# 自定义 Kernel 与系统收官实验

```{contents} 本页目录
---
depth: 2
local: true
---
```

前面的实现章节已经说明 Attention Backend、CUDA Graph 与 Tensor Parallel 怎样调用底层算子。本章继续追到 Python 入口之后的 C++、CUDA 与 Triton 实现，按 shape、stride 和 index 理解内存访问，并区分通用算子与值得定制的高频窄路径。

本章依据仓库提交 `9a91cfafe754aa85daee49998176275667eb58f2`。调用关系和算子语义已经逐项核对；带宽、延迟与加速比没有在 GPU 上实测。涉及性能的结论只表示设计意图、源码推导或待验证假设。

## 热点形状决定性能路线

Mini-SGLang 按工作位置和数据形状选择实现。共同目标是减少 Python 调度、临时 tensor 和重复 launch，让一次调用完成更多有效工作。

| 路线 | 对象 | 适用原因 | 主要代价 |
|-|-|-|-|
| CPU C++ | TensorMatcher、radix 比较 | TensorMatcher 在 host 侧校验 TensorView 元数据；radix 比较的数据和控制状态已在 CPU，可避免 H2D 与 GPU launch。 | 首次加载要编译；短输入可能受 FFI 开销支配。 |
| CUDA C++ JIT | indexing、store_cache | 整行搬运适合 warp 向量化，也能把 mask 或 K/V 双写并入一次 launch。 | 模板组合增加编译缓存，输入必须满足行宽、对齐和索引契约。 |
| Triton JIT | MoE GEMM、top-k reduce | 按专家整理后的稀疏计算可变成规则 tile，用 tl.dot 表达块矩阵乘。 | 配置依赖 shape 和 GPU，路由及激活仍依赖外部 kernel。 |

PyNCCL 是同一基础设施的通信消费者：它复用 TVM FFI、TensorView 和当前 CUDA stream，把 tensor parallel collective 接到 NCCL。

## TVM FFI 连接 Python 与本地 Kernel

### AOT 加载源文件，JIT 实例化模板

公共入口位于 python/minisgl/kernel/utils.py。load_aot 调用 tvm_ffi.cpp.load，编译 csrc/src 中的 tensor.cpp、radix.cpp 或 pynccl.cu。这里的 AOT 是相对 inline JIT 而言，模块仍在第一次调用时按需编译，并非安装 wheel 时预编译。

load_jit 调用 tvm_ffi.cpp.load_inline。它包含 csrc/jit 下的 CUDA 文件，再生成 TVM_FFI_DLL_EXPORT_TYPED_FUNC 包装，把模板实例导出成 launch。index 和 store 因而能把行字节数、线程数与 PDL 开关固化为编译期常量。两条路径都使用 C++20 和 -O3，CUDA 额外启用 --expt-relaxed-constexpr。

```text
Python wrapper
  → functools.cache 模块加载函数
  → load_aot：编译 csrc/src/*.cpp 或 *.cu
     或 load_jit：include csrc/jit/*.cu 并实例化模板
  → TVM FFI 导出函数
  → C++ 接收 tvm::ffi::TensorView
```

### Launch 使用 TVM FFI 提供的 CUDA stream

LaunchKernel::resolve_device 通过 TVMFFIEnvGetStream 取得框架提供的 stream，再由 cudaLaunchKernelEx 启动 kernel。源码设计让扩展 kernel 跟随该 stream 提交；它与 PyTorch 当前 stream 的一致性以及进入 CUDA Graph 捕获后的行为，仍需在 CUDA 环境用多 stream 和 graph replay 实验验证。若误用 legacy default stream，异步执行可能读到尚未完成的数据。

### TensorMatcher 负责入口契约

python/minisgl/kernel/csrc/include/minisgl/tensor.h 定义 SymbolicSize、SymbolicDType、SymbolicDevice 和 TensorMatcher。符号第一次校验时绑定，后续参数复用同一符号，便能表达维度、dtype、device 必须一致。未声明 stride 时要求 contiguous；显式给出 stride 后逐维验证。

tensor.cpp::test 是机制自测。它要求 x 为 [N,1024] 的 CPU tensor，末维 stride 为 1；y 为同一 N 的 contiguous CUDA int32 或 int64 tensor，并要求 N 能被 4 整除。测试脚本把 y 固定在 cuda:1，至少需要两张可见 GPU。

<callout emoji="💡">
**契约缺口：**SymbolicSize 用 0 表示“尚未绑定”，零长度维度没有可靠语义。tensor.py 又把返回类型标成 int，而 C++ 函数返回 void。现有正例不能证明空 batch 已受支持。
</callout>

## Indexing 融合 Embedding Gather 与 TP Mask

```text
VocabParallelEmbedding.forward
  → minisgl.kernel.indexing
  → _jit_index_module(element_size, num_splits)
  → IndexKernel<...>::run
  → index_kernel 或 masked_index_kernel
```

python/minisgl/kernel/index.py 默认创建 [L,D] 输出，element_size 等于 D 乘 dtype 字节数。行宽能被 2048 整除时，一行由 4 个 warp 搬运；否则能被 1024 整除时用 2 个 warp，其余用 1 个 warp。split 只切分行内字节，不改变数值语义。

普通 kernel 读取 indices[token]，从 weights 的随机行取一条 embedding，再连续写入 output。TP vocabulary 模式额外接收 start 和 length：本 rank 范围内读取局部权重，范围外直接清零，最后各 rank all-reduce。索引平移、mask、gather 和清零由一次 launch 完成。

默认 block 有 128 个线程，即 4 个 warp。不同 token 的源行通常不连续，但同一 warp 的 32 个 lane 访问一行中的相邻地址，行内读写可以合并。masked kernel 将 index-start 作为无符号数比较，低于 start 的负差值会变成大整数并进入清零分支。

<callout emoji="💡">
**风险：**普通 indexing 不检查 index 上下界，负数或越界值可能触发非法显存读取。每个 warp 负责的字节数必须是 128 的倍数，Python 包装没有提前检查。
</callout>

## Store Cache 用一次 Launch 写入 K 和 V

FlashAttention、FlashInfer 和 TensorRT-LLM backend 都会在 attention 前调用 MHAKVCache.store_kv。cache 被视为 [num_token_slots, local_kv_heads × head_dim]；scheduler 根据 page table 生成 batch.out_loc，作为目标 slot。

```text
BaseAttnBackend.forward（由具体 backend 实现）
  → MHAKVCache.store_kv
  → store_cache
  → _jit_store_module(element_size)
  → StoreKernel<...>::run
  → store_kv_cache
```

一个 warp 处理一个输入 token：读取 pos=indices[token]，把 K 行写到 k_cache[pos]，再把 V 行写到 v_cache[pos]。两次 scatter 共用一次 launch，也不需要高级索引创建中间 tensor。cache 与输入可以有不同的行 stride，方便接收从融合 QKV tensor 切出的 view；末维必须连续。

<callout emoji="💡">
**风险：**store_cache 不检查 index 边界。重复 index 会让多个 warp 并发写同一 cache 行，形成数据竞争；现有测试明确使用无重复索引。行字节数和对齐仍受 warp::copy 约束。
</callout>

## Warp 工具把行搬运变成合并访存

python/minisgl/kernel/csrc/include/minisgl/warp.cuh 按总字节数选择 16 字节 uint4、8 字节 uint2 或 4 字节 uint1。一个 warp 每轮分别搬运 512、256 或 128 字节；lane i 处理第 i 个向量包，循环次数是模板常量并会展开。

```cpp
lane_id = threadIdx.x % 32;
for (i = 0; i < kLoopCount; ++i) {
  j = i * 32 + lane_id;
  dst_packed[j] = src_packed[j];
}
```

它不用 shared memory，也没有 warp 外同步。代价是总字节数必须为 128 的倍数，地址还要满足向量类型的对齐。PyTorch allocator 通常能提供良好对齐，带偏移的 view 仍需验证。

KernelConfig 的 max_occupancy 最终进入 __launch_bounds__(threads,value) 第二参数。CUDA 对它的定义是“每个 SM 至少可驻留多少个 block”的编译提示，并非最大并发数。真实 occupancy 要结合寄存器、block 大小和 profiler 判断。

例如每行 2048 字节、使用 uint4 搬运，一个 Warp 的 32 个 Lane 每轮共搬 512 字节，需要 4 轮。Lane i 在每轮处理第 i 个 16-byte 包；这保证行内地址连续，但不同 token 的源行仍可能离散。此类 Kernel 的优化目标是高效搬运与减少 launch，并未增加数学计算量。

## Radix 比较留在 CPU 避开启动成本

请求进入 prefill 后，CacheManager.match_req 沿 RadixPrefixCache._tree_walk 查找前缀，RadixTreeNode.get_match_len 最终调用 fast_compare_key。radix.cpp 要求参数是一维、contiguous、同 dtype 的 CPU int32 或 int64 tensor。

实现对公共长度调用 std::mismatch，返回首个不同元素的位置；公共部分完全相同时返回较短序列长度。这就是最长公共前缀。Python 随后按 page_size 向下对齐，必要时拆分 radix 节点。

token IDs、树节点和 LRU 状态本来就在 CPU。为短整数序列发起 H2D 和 GPU launch，成本很可能高于比较本身；C++ 的收益主要来自摆脱 Python 逐元素循环。

<callout emoji="💡">
**待验证：**仓库没有 fast_compare_key 专项性能测试。短 key 可能受 FFI 开销支配；长 key 且差异靠后时更可能受益。
</callout>

## Triton 把 MoE 路由整理成块矩阵乘

```text
MoEMLP.forward：生成 router logits
  → FusedMoe.forward
  → sgl_kernel.topk_softmax
  → sgl_kernel.moe_align_block_size
  → Triton expert GEMM #1
  → FlashInfer silu_and_mul 或 gelu_and_mul
  → Triton expert GEMM #2
  → Triton top-k sum reduce
  → TP all-reduce
```

仓库的 fused MoE 由多段 kernel 组成。自研部分是两次专家 GEMM 和最终 reduce；top-k、路由对齐与激活依赖 sgl_kernel 或 FlashInfer。

moe_align_block_size 把 [M,top_k] 的专家 ID 展平并按 expert 分组，再把各专家 route 数补齐为 BLOCK_SIZE_M 的倍数。sorted_token_ids 保存 route ID，expert_ids 指定每个 M block 的专家。同一个 tile 内的行因而共享一份专家权重。

fused_moe_kernel 的 grid 等于 M block 数乘 N block 数。GROUP_SIZE_M 重排 program ID，争取复用 L2 中的专家权重。每个 program 通过 route_id/top_k 找回 token，按 expert_id 选择 B[expert]，沿 K 维 tl.load，并用 tl.dot 累加到 FP32。K、N 尾块和 padding route 由 mask 保护。

第一次 GEMM 后执行门控激活，第二次 GEMM 把 route 输出投回 hidden size。router weight 只在两次 GEMM 中的一次乘入。moe_sum_reduce_kernel 再沿 top-k 以 FP32 累加，转回输入 dtype；当前固定 BLOCK_M=1、BLOCK_DIM=2048、num_warps=8。

### Triton 源码的阅读抓手

- @triton.jit 按 constexpr 参数和输入类型生成 GPU kernel。
- tl.program_id 选择 tile，角色接近 CUDA block ID；tl.arange 生成向量化地址。
- tl.load 和 tl.store 的 mask 处理边界；tl.dot 表达 tile GEMM。
- tl.dot 使用何种 Tensor Core 指令取决于 dtype、架构和编译结果，需要查看产物或 profiler。

### 这里的 fused 到底指什么

**它不是把整条 MoE 前向压成一个 GPU kernel。**这个 Triton kernel 融合的是一次专家线性层里的四件事：根据路由结果收集输入行、按 expert 选择权重、执行分块矩阵乘、把结果散射回原 route 槽位；还可以在写回前乘一次 router weight。

因此，仓库里的 fused MoE 更准确地说是一条由多个 kernel 组成的流水线。top-k、按 expert 分组、第一次专家 GEMM、门控激活、第二次专家 GEMM 和 top-k 求和仍然是独立步骤。这里的 fused_moe_kernel 只负责其中一次 expert GEMM，但它省掉了显式构造“按 expert 重排后的输入矩阵”和显式反向 permutation 的中间 kernel。

```text
r = 展平后的 route ID，取值范围 [0, M × K_top)\nt(r) = floor(r / K_top)\ne(r) = topk_ids.flatten()[r]\n\nC[r, n] = α(r) × Σ_k A[t(r), k] × B[e(r), n, k]\nα(r) = topk_weights.flatten()[r] (当 MUL_ROUTED_WEIGHT=True)，否则为 1
```

### 用 Shape 账本读懂两次调用

与第 9 章符号保持一致：设本轮输入 token 数为 `M`，每个 token 选出 `K_top` 个专家，有效路由总数为 `R = M × K_top`，隐藏维度为 `H`，本 rank 的单专家中间维度为 `I_local = I / P`。`w1` 合并了 gate 与 up 投影，输出维为 `2I_local`。

| 阶段 | 输入 A | 权重 B | 输出 C | 传入 `top_k` |
|-|-|-|-|-|
| 第一次专家 GEMM | `[M, H]` | `[E, 2I_local, H]` | `[M, K_top, 2I_local]` | `K_top` |
| 门控激活 | `[R, 2I_local]` | 无 | `[R, I_local]` | 无 |
| 第二次专家 GEMM | `[R, I_local]` | `[E, H, I_local]` | `[M, K_top, H]` | `1` |
| 专家加权归约 | `[M, K_top, H]` | 无 | `[M, H]` | 无 |

最关键的细节是第二次调用把参数 `top_k` 显式传为 `1`：第一次调用时 `A` 仍是原始 token 矩阵 `[M, H]`，因此路由槽位 `r` 要通过 `r // K_top` 找回所属 token 行；门控激活后每条路由已独立占一行 `[R, I_local]`，第二次调用直接按 `A[r]` 读取即可。

router weight 只乘一次。apply_router_weight_on_input 为真时，第一次 GEMM 的 MUL_ROUTED_WEIGHT 为真；否则第二次为真。这样无论选择哪种时机，最终每条 route 都只带一个路由权重。

### 逐段拆解 fused_moe_kernel

#### program ID 如何定位输出 tile

launch grid 是 M 方向块数乘 N 方向块数。一个 Triton program 负责 [BLOCK_SIZE_M, BLOCK_SIZE_N] 的输出 tile。扁平 pid 先被还原成 pid_m 和 pid_n，再由它们确定 route 行块与输出列块。

GROUP_SIZE_M 只改变 program 的访问顺序，不改变结果。组内先遍历一批相邻的 M block，再切换 N block。路由已经按 expert 排序，相邻 M block 往往属于同一个专家，因此它们读取的 B 权重块更容易留在 L2 中。最后一组不足 GROUP_SIZE_M 时，group_size_m 会收缩，避免生成越界的 pid_m。

#### 为什么同时需要 R、P 和 EM

| 符号 | 对应变量 | 含义 |
|-|-|-|
| R | num_valid_tokens | 真实 route 槽位数，即 topk_ids.numel() = M × T |
| P | num_tokens_post_padded | 每个 expert 的 route 数补齐到 BLOCK_SIZE_M 后的实际长度 |
| EM | sorted_token_ids.shape[0] | sorted_token_ids 的预分配容量，也是 grid 采用的上界 |

三者满足 R ≤ P ≤ EM。grid 按 EM 启动，避免先把设备端的 P 同步回 CPU；每个 program 再读取 P，如果 pid_m × BLOCK_SIZE_M 已经越过 P 就直接返回。进入有效块后，sorted_token_ids 中的补齐位置会写入不小于 R 的哨兵 route ID，token_mask = offs_token < R 会让这些行既不参与有效 load，也不写回 C。

#### A 是 gather，B 是按 expert 寻址

`offs_token_id` 是按专家排序后的位置，而从 `sorted_token_ids` 取出的 `offs_token` 才是原始路由槽位 ID。读取 `A` 时使用 `offs_token // top_k`，因此第一次调用会让同一 token 的 `K_top` 条路由共享同一行 hidden state，第二次调用令 `top_k=1` 直接读取激活后的路由行。

expert_ids[pid_m] 给出整个 M block 对应的专家。moe_align_block_size 已经保证同一个 M block 内的 route 属于同一 expert，所以 B 只需加一次 off_experts × stride_be，就能让整个 tile 共享该专家的权重。随后，offs_k 与 offs_bn 分别展开 K 行和 N 列，形成 [BLOCK_SIZE_K, BLOCK_SIZE_N] 的 B 指针块。

N 的尾块使用取模后的 offs_bn 读取 B，避免构造越界地址；真正写回时仍使用未取模的 offs_cn，并通过 offs_cn < N 屏蔽多余列。因此尾块中被环绕读取的值不会落到输出。

#### K 循环、路由权重与 scatter

accumulator 固定用 FP32。kernel 沿 K 维每次推进 BLOCK_SIZE_K，分别加载 A tile 和 B tile，再用 tl.dot 累加。even_Ks 是编译期常量：K 能整除 BLOCK_SIZE_K 时省掉 K 尾块 mask，否则只在最后一个 K tile 屏蔽越界元素。route padding 则始终由 token_mask 处理。

矩阵乘完成后，MUL_ROUTED_WEIGHT 决定是否加载 topk_weights.flatten()[offs_token] 并逐行相乘。结果随后转换为 compute_type。

写回地址直接使用原始路由 ID `offs_token`。对于连续输出张量 `C=[M, K_top, N]`，传入的 `stride_cm` 等于 `C.stride(1)=N`，`stride_cm × offs_token` 把前两维视作展平的路由维 `R`，在写回的同时完成逆向 scatter，省去额外的重排 kernel。

## PyNCCL 的 Kernel 边界

通信初始化、collective 路径和系统实验见分布式章节。本章只保留 kernel 契约：pynccl.py 通过 load_aot 编译 pynccl.cu 并链接 libnccl；all-reduce 支持 fp16、bf16，消息大小不超过 `m_max_bytes` 时可先复制进 NCCL symmetric buffer，collective 后再复制回原 tensor，大消息直接原地调用；all-gather 直接写最终输出。两者都从 TensorView 解析 stream。test_comm.py 固定 4 张 GPU，只覆盖 fp16 与一个主要消息规模。

## 测试口径限定了结论范围

| 脚本 | 已覆盖 | 缺口 |
|-|-|-|
| test_index.py | batch 1 到 32768；对比 F.embedding；含 TP mask。 | 越界、非对齐 view、多 dtype 与行宽。 |
| test_store.py | 对比 torch.compile 的 indexed assignment。 | 重复或越界 index、非 128 字节行宽。 |
| test_tensor.py | 一个合法 TensorMatcher 输入。 | 失败诊断、空维度；脚本要求 cuda:1。 |
| test_comm.py | 4 GPU；all-reduce 正确性与延迟，all-gather 正确性。 | 拓扑、dtype、消息大小和阈值扫描。 |
| MoE | 没有专项测试。 | 正确性、精度、padding 浪费与性能均未验证。 |

benchmark/perf.py 先 warmup，再把多次调用捕获进 CUDA Graph，用 CUDA Event 测 replay 平均时间。它适合稳定态比较，不反映首次编译。报告的 GB/s 是有效 payload 除以耗时：index 未同时计入源读和目标写，store 也未把读写两侧都加进去，只能用于同口径相对比较，不能直接当作 DRAM 利用率。

## 从单算子契约走向全系统消融实验

单个 Kernel 的微基准（如 `benchmark/perf.py`）只能衡量固定形状下的稳态搬运或矩阵乘耗时，并且依赖 Linux + CUDA 12.8+ 编译环境（依赖锁定与安装运行命令统一收在附录 A）。要把前八章走读过的调度、缓存、TP、Attention、CUDA Graph 与自定义算子真正闭合，还必须在统一服务口径下做单变量消融实验。

## 度量口径沿用前文 Benchmark

客户端指标沿用第 4 章〈[推理基准：延迟、吞吐、Goodput 与观测边界](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyY6hn6oMy0X2hIG4DWlU7dg)〉的统一口径。需注意仓库自带的 `benchmark/client.py` 采用了不同计时点：它在拿到流式响应对象后才记录起点，并为每个 SSE 块（含空块与结束块）打点，因此其“TTFT”漏算了建连与首包头等待，“TPOT”实为块间间隔，`token/s` 分子还混入了起始时间戳，仅适合该脚本内部相对对比，不能跨工具横向引用。

| 测量目标 | 需保存的原始观测数据 | 统一计算方式 |
|-|-|-|
| 用户首包等待 | 发起 `create` 前的单调时钟；首个非空内容块时刻 | 二者之差记为客户端 TTFT；仅含角色或空内容的块另列 |
| 后续生成速度 | 各非空内容块到达时刻；实际输出 token 总数 | 严格按第 4 章内容完成口径计算 E2E 与 TPOT；块间 ITL 独立命名 |
| 稳态吞吐 | 正式测量窗口时长、成功/失败数与真实输出 token 数 | 分子与窗口严格对齐，剔除起始时间戳与 `[DONE]` 终止帧 |

修改计时口径后要重新建立基线。原客户端逐块打点的数据不能恢复已经漏掉的请求前段等待，也不能在没有实际 token 数时恢复严格的逐 token 统计。

<callout emoji="📏">
**本章原则：**先过正确性门槛，再比较性能；每次只改变一个因素，同时保留 workload、模型、硬件、软件版本、warmup、缓存状态和原始结果。
</callout>

## 核心消融矩阵

| 问题 | 控制变量 | 重点观察 | 正确性门槛 |
|-|-|-|-|
| Overlap 是否隐藏 CPU 开销 | 默认与 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` | TPOT、GPU idle gap、完成 token 数 | 输出长度严格一致，只出现一次 finished |
| CUDA Graph 是否减少 launch overhead | `--cuda-graph-max-bs 0` 与默认 | decode TPOT、metadata/plan 与 replay 时间 | logits、token、KV 与 eager 在容差内一致 |
| Radix 是否复用共享前缀 | `--cache-type radix` 与 `naive`，冷/热 prefix | 实际 extend token、TTFT、可驱逐页 | 输出相同，命中位置页对齐且保留末 token forward |
| Chunked Prefill 如何改变峰值 | 不同 `--max-prefill-length` | 峰值显存、TTFT、decode 抖动 | 各 chunk 复用同一 table row，中间 chunk 无用户输出 |
| Attention backend 是否等价 | `fa`、`fi`、`trtllm`、`fa,fi` | prefill/decode latency、workspace、page metadata | 固定 greedy 下逐 token 一致，随机采样比较分布 |
| TP 是否值得 | TP=1/2/4，PyNCCL 与 torch NCCL | 吞吐、通信占比、显存、扩展效率 | 各 rank batch UID 相同，collective 顺序一致 |

## 用 Nsight 拆开一轮 decode

一次 decode 的总时间至少包含 scheduler 准备、metadata 或 FlashInfer plan、静态 buffer copy、CUDA Graph replay 或 eager launch、sampling、D2H copy 和 CPU 后处理。性能分析时应分别打点，而不是把整条加速都归因于 CUDA Graph。

- Nsight Systems：观察 CPU 提交、两条 CUDA stream、NCCL collective、空闲间隙与 overlap。
- Nsight Compute：观察 index/store kernel 的 global load/store efficiency、DRAM throughput、warp stall 和 occupancy。
- `NCCL_DEBUG=INFO`：核对初始化、拓扑和可见错误；逐轮 collective 顺序还需结合 trace 或显式打点，INFO 日志本身不保证记录每次调用。
- 服务日志：记录每轮 batch UID、phase、真实/补齐 batch size、extend token 数、cache hit 与完成原因。

## 用生产标准审视 Mini-SGLang 的边界

Mini-SGLang 的价值在于把核心机制写得短而直接。它的简化也意味着“能展示生产机制”与“已经达到生产要求”之间仍有距离。下面的边界全部来自当前源码静态核对；带“待验证”的项还需要运行证据。

### 模型、缓存与调度边界

- 注册表只覆盖 Llama、Qwen2、Qwen3、Qwen3-MoE、Mistral/Mistral3 文本路径；没有通用 AutoModel fallback。
- Mistral 的 sliding-window 配置没有进入 backend，因此不能据此宣称完整复现对应语义。
- KV pool 只实现 `MHAKVCache` 这一类 K/V 分离缓存池，可承载 MHA、GQA 和 MQA 的 head 布局；MLA、HiCache、host/offload 与多级缓存仍未实现。
- Radix `reset()` 未实现，树的 `check_integrity()` 为空；现有完整性检查主要覆盖页数算术与 free slot 对齐。
- 调度固定 prefill-first，没有抢占、swap、recompute、优先级、quota 或显式抗饥饿机制。
- Chunked Prefill 降低单轮 extend 的计算与激活峰值，但 admission 仍预留完整未来 KV，不能让超过总 KV 容量的请求运行。

### 收官验收

- 提交一份包含 workload、硬件、软件版本、warmup、指标定义、原始数据和结果解释的 benchmark 记录。
- 每个消融先过正确性门槛，再比较性能，不能用更快掩盖输出错误。
- 至少注入客户端断连、超长 prompt、一个 rank 退出和 KV 容量紧张四类故障。
- 把发现分成正确性、资源安全、性能、接口兼容与可操作性，并标明证据等级。
- 能说明哪些思想可以迁移到 SGLang/vLLM，哪些只是 Mini-SGLang 当前实现选择。

这些实验把静态判断变成可复现证据。完成本章后，应能固定 workload 与环境，先过正确性门槛，再用 TTFT、TPOT、吞吐、显存与 timeline 判断瓶颈；仍未运行的结论继续标为待验证。下一章以这些边界为坐标，对照生产版 SGLang 的能力与源码。

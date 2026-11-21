# Attention Backend 把 Q/K/V 接入 Paged KV Cache

```{contents} 本页目录
---
depth: 2
local: true
---
```

读到这一层时，不必再回顾 Transformer 的基础公式。这里要解决的是一个工程问题：一批长度不同、命中前缀不同的请求，怎样在同一轮 forward 中变成 kernel 能消费的 Q、页表和 KV Cache。读完本节，你应该能从 **Scheduler._prepare_batch** 一路追到三个 attention backend，写出关键张量的 shape，并说明 prefill、decode、CUDA Graph 复用的是哪些数据。

<callout emoji="💡">
**先记住主线：**模型层生成并旋转 Q/K，backend 把本轮 K/V 写入统一 KV Cache，再用各自的 paged-attention API 读取完整上下文。三个 backend 的差别集中在 metadata 布局和 kernel 调用，不在模型语义。
</callout>

## 三层职责把模型语义和 kernel 细节隔开

| 层次 | 主要符号 | 职责边界 |
|-|-|-|
| 模型与 Attention Layer | **RopeAttn**、**AttentionLayer** | 完成 fused QKV projection、可选 Q/K Norm、RoPE、head reshape，并把结果交给全局 backend。它不分配 KV 地址，也不拼 kernel metadata。 |
| Attention backend | **BaseAttnBackend**、FA、FI、TRTLLM | 把 Batch 翻译成后端元数据；按 **out_loc** 写入当前层 KV；选择 prefill 或 decode kernel；为 CUDA Graph 准备稳定地址。 |
| 调度与缓存管理 | **Scheduler**、**CacheManager**、**GraphRunner** | 决定本轮请求和 phase，分配物理页，生成 positions、input mapping、out_loc，并在需要时补 dummy request 到 graph bucket。 |

**BaseAttnBackend** 暴露五组动作：**forward**、**prepare_metadata**，以及 CUDA Graph 的 **init_capture_graph**、**prepare_for_capture**、**prepare_for_replay**。**HybridBackend** 只按 phase 路由：prefill 走第一个 backend，decode 走第二个；图捕获只委托 decode backend，因为 GraphRunner 从不捕获 prefill。

**一轮 forward 的真实调用顺序如下：**

```text
Scheduler._prepare_batch
  allocate_paged(real requests)
  positions = [cached_len, device_len)
  out_loc = page_table[request_row, positions]
  backend.prepare_metadata(batch)

RopeAttn.forward
  qkv = qkv_proj(hidden_states)
  q, k, v = split(qkv)
  q_norm(q); k_norm(k)          # 仅部分模型启用
  q, k = RoPE(positions, q, k)
  output = backend.forward(q, k, v, layer_id, batch)

backend.forward
  store_kv(rotated_k, v, out_loc, layer_id)
  read paged K/V cache and run attention
```

这个顺序带来两个直接结论。第一，K Cache 保存的是做过 RoPE 的 K，V 不参与旋转。第二，当前 token 的 K/V 先写入 cache，再参与本轮 attention。Prefill 会一次写入整个 extension，causal kernel 负责遮住同一 extension 中位于当前位置之后的 token。

## Shape ledger 是排查后端问题的第一张表

下面统一使用这些符号：真实 batch size 为 **B**，CUDA Graph padding 后为 **B_p**；第 i 个请求本轮计算长度为 **e_i = device_len - cached_len**，完整上下文长度为 **s_i = device_len**；**T = Σe_i**；模型全局的 Q、KV head 数分别为 **Gq**、**Gkv**，当前 TP rank 上的 Q、KV head 数分别为 **Hq**、**Hkv**；head dimension 为 **D**；物理 page size 为 **P_page**。

| 对象 | shape | 含义与约束 |
|-|-|-|
| hidden states | `[T, hidden_size]` | 不同请求的 extension 按请求顺序连续打包，不补成矩形 token 张量。 |
| fused QKV | `[T, (Hq + 2Hkv) × D]` | Q、K、V 已按当前 TP rank 切分；K/V head 可以少于 Q head。 |
| Q | `[T, Hq, D]` | AttentionLayer 在调用 backend 前显式 reshape。 |
| K、V 输入 | `[T, Hkv × D]` | store kernel 把尾部视为一段连续向量；K 已做 RoPE。 |
| 单层 K 或 V Cache | `[num_pages, P_page, Hkv, D]` | 写入时展平成 `[num_pages × P_page, Hkv, D]`，out_loc 直接索引第一维。 |
| positions、out_loc | `[T]` | positions 是逻辑绝对位置；out_loc 是物理 token slot，两者不能混用。 |
| cu_seqlens_q、cu_seqlens_k | `[B_p + 1]` | 前缀和数组；最后一个元素分别是 packed Q 数和所有请求 KV 长度之和。 |
| attention output | `[T, Hq, D]` | 随后展平成 `[T, Hq × D]`，交给 row-parallel o_proj。 |

## Scheduler 先决定物理位置，再让 backend 组 metadata

### cu_seqlens_q 有三种构造分支

**Req.cached_len** 表示本轮开始前已经可读的 KV 前缀，**device_len** 表示纳入本轮新 token 后的上下文终点，**extend_len** 是二者之差。每次 forward 完成后，**complete_one** 先把 cached_len 推到旧 device_len，再把 device_len 加一，于是下一轮 decode 的 extend_len 恰好为 1。

| 分支 | 判定 | cu_seqlens_q | 用途 |
|-|-|-|-|
| 所有 query 长度均为 1 | `max(e_i) == 1` | `[0, 1, ..., B_p]` | 包括 decode，以及每个请求都只剩一个新 Q 的 prefill；graph 路径还包含 padding 的 dummy request。 |
| 无 prefix hit 的 prefill | 所有 `cached_len == 0` | 直接复用 `cu_seqlens_k` | 每个请求的 Q 长度等于完整 KV 长度，省去一次构造和拷贝。 |
| 部分命中或 chunked prefill | 其他情况 | `cumsum([0] + e_i)` | Q 只包含新算的 extension，K/V 上下文还包含已缓存前缀。 |

这里的分支只决定 `cu_seqlens_q` 怎样构造。FI、TRTLLM 等 backend 选择 prefill 或 decode wrapper 时，仍以 `Batch.phase` 为准。

```text
无命中：e=[4,2], s=[4,2]
  cu_q=cu_k=[0,4,6]，last=[3,5]
部分命中：cached/device=[(3,5),(1,4)]
  e=[2,3]，cu_q=[0,2,5]，cu_k=[0,5,9]，last=[1,4]
decode：cached/device=[(5,6),(8,9)]
  cu_q=[0,1,2]，cu_k=[0,6,15]，last=[0,1]
```

**get_last_indices** 返回 `cu_q[1:B+1] - 1`。LM head 在 prefill 时只取这些 packed Q 的末端位置，因此这里返回的是本轮计算结果的下标，不是 KV Cache 的绝对地址。

### page table 保存 raw token slot

全局 page table 的每个元素都是物理 token slot。CacheManager 以 page 为单位分配，再把每个 page 展开为 P_page 个连续 token 地址写入表中；prefix cache 命中时则把已有地址复制进新请求的表行。Scheduler 用二维索引 `(table_idx, position)` 查出 out_loc，所以 store kernel 无需理解逻辑请求、page 编号或 radix tree。

### 页表就绪后再启动 Attention

CPU 和 GPU 各维护一份 Block Table，是一种可行的实现方式。对于两块独立分配的缓冲区，程序需要显式传递更新，并用 CUDA stream 或 event 约束执行顺序：本轮 Attention 开始读取前，对应的块表和长度必须已经就绪。CPU 可以提前准备下一批，两端内容允许暂时不同。

Mini-SGLang 采用另一种组织方式：全局 `Engine.page_table` 直接放在 GPU 上，CPU 上的 Python 对象持有这个 CUDA Tensor 的引用，没有另存一份完整 CPU 页表。理解一致性时，需要分别看双份表怎样同步，以及这个实现怎样发布本批 metadata。

#### 双份块表按批次发布

以页大小为 16 的单请求为例。已有 32 个 KV token，块表为 `[12, 7]`；本轮要写入第 33 个 token，调度器再分配物理页 42，准备块表 `[12, 7, 42]` 和有效长度 33。CPU 先确定这组数据，再把块号与长度传给 GPU。这次传输只涉及 metadata，历史 K/V 仍留在原物理页中。

若拷贝和计算提交到同一条 CUDA stream，队列顺序保证先复制、后读取。使用不同 stream 时，可以在拷贝之后记录事件，让计算 stream 等待事件。下面是时序示意，省略了缓冲区分配和模型实现；进入示例前，CPU 源缓冲区已经准备好，GPU 目标缓冲区也已不再被上一批使用。

```python
with torch.cuda.stream(copy_stream):
    block_table_gpu.copy_(block_table_cpu, non_blocking=True)
    seq_lens_gpu.copy_(seq_lens_cpu, non_blocking=True)
    metadata_ready.record()

with torch.cuda.stream(compute_stream):
    compute_stream.wait_event(metadata_ready)
    write_current_kv(...)
    attention(...)
```

`wait_event()` 为计算 stream 的后续工作建立依赖，CPU 可以继续提交任务。`non_blocking=True` 允许拷贝异步提交，不表示数据已经到达 GPU。块表、长度和相关索引都需要在记录事件前准备完，Attention 才不会读到“新长度配旧块表”。

#### 异步复制要保护缓冲区的使用期限

先更新、后读取解决了本批数据的可见性。流水线还要处理另一方向的依赖：旧操作没有结束时，下一批不能提前复用它正在访问的存储。

| 对象 | 过早复用的后果 | 需要满足的条件 |
|-|-|-|
| CPU pinned 源缓冲区 | H2D 仍在读取时改写，GPU 可能收到混合数据。 | 等待本次拷贝完成后再改写，或使用独立源缓冲区。 |
| GPU metadata 缓冲区 | 上一批 kernel 尚未读完，表项已被下一批覆盖。 | 等待旧读取结束后再覆盖，或为在途批次保留独立缓冲区。 |
| 物理 KV 页 | 旧请求仍在访问，页中内容已被新请求覆盖。 | 引用与回收规则保护仍需使用的页；复用后的写入必须排在旧 GPU 访问之后。 |

在这类设计中，调度器统一决定地址映射，Attention 消费本批映射并读写 K/V。常规计算不需要 GPU 反向修改块表，再把修改同步回 CPU。若实现允许多方更新映射，还需另外设计写入协调规则。

#### Mini-SGLang 直接更新 GPU 页表

`Engine` 创建页表时明确指定 `device=self.device`，其中 `self.device` 是当前 rank 的 CUDA 设备。`TableManager` 和 `CacheManager` 接收同一个页表 Tensor；`CacheManager.free_slots` 也位于相同 GPU。调度决策由 CPU 执行，并不要求页表数值保存在 CPU 内存里。

`allocate_paged()` 在 CPU 上计算每条请求需要新增的逻辑页区间，并选取 GPU 空闲页索引。`_write_page_table()` 随后在 pinned memory 中准备请求行号和逻辑位置，将这两组索引传到 GPU，再把已经位于 GPU 的 `allocated` 写入页表：

```python
table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
offsets = positions_host.to(page_table.device, non_blocking=True)
page_table[table_idxs, offsets] = allocated
```

这里传输的是“更新哪一行、哪些列”的索引。页表更新后，`_prepare_batch()` 先取出本批 `out_loc`，再调用 `prepare_metadata()`。FlashAttention 用 `torch.stack()` 生成本批独立块表，FlashInfer 用 `torch.cat()` 生成本批独立 indices；两者都从 GPU 全局页表取值。具体页粒度转换由下一节展开。

#### 计算 stream 等待本批 metadata 就绪

开启 overlap 时，页表写入、`out_loc` 提取与 `prepare_metadata()` 均提交在 `scheduler stream`；进入模型前，`engine stream` 通过 `wait_stream()` 建立单向等待，保证每层 `store_kv()` 与分页读取看到的都是已更新的本批块表。双流接力与 CPU pinned 缓冲保护的完整时序在下一章统一展开。

## 统一 KV Cache 支撑三种 page 解释

**MHAKVCache** 的总 storage 是 `[2, num_layers, num_pages, P_page, Hkv, D]`。每层 backend 都先调用同一个 **store_kv**，JIT CUDA kernel 让一个 warp 处理一个 token，根据 out_loc 把完整 K 向量和 V 向量复制到目标 slot。读路径随后分成两种 page 解释：

- **FA 与 TRTLLM：**从 raw token table 每隔 P_page 个元素取一个 page 起点，再整除 P_page，得到矩形 block table，shape 约为 `[B_p, ceil(max(s_i) / P_page)]`。
- **FI：**把 cache reshape 成 `[num_pages × P_page, 1, Hkv, D]`，将每个 token 当成 page size 为 1 的逻辑页；indices 是各请求 raw token slot 的一维拼接。

这两种解释指向同一片物理显存。FA/TRTLLM 的 metadata 随 block 数增长；FI 当前实现的 indices 随上下文 token 数增长。比较后端 CPU 开销时，不能只看 attention kernel 时间，还要把这部分 metadata 规划和 H2D 拷贝算进去。

## 三个 backend 的差别落在 metadata 与 kernel

| 比较项 | FlashAttention（fa） | FlashInfer（fi） | TensorRT-LLM FMHA（trtllm） |
|-|-|-|-|
| 运行入口 | `sgl_kernel.flash_attn.flash_attn_with_kvcache` | FlashInfer prefill/decode wrapper | FlashInfer 暴露的 TRTLLM context/decode 函数 |
| phase | prefill、decode 共用统一入口 | 两个 wrapper，按 batch phase 选择 | context 与 decode 两个函数 |
| KV 视图 | `[N,P_page,Hkv,D]` | `[N×P_page,1,Hkv,D]` | `[N,P_page,Hkv,D]` |
| 页表 | 矩形 block table | ragged token-slot indices | 矩形 block table |
| 规划数据所在设备 | 主要为 GPU int32 | 长度和 indptr 主要放 pinned CPU，indices 与 Q offsets 在 GPU | 主要为 GPU int32 |
| RoPE | 进入 backend 前已完成 | `pos_encoding_mode=NONE`，避免重复旋转 | 进入 backend 前已完成 |
| scale | 显式传 `D^-0.5` | Python 层未显式传，依赖 wrapper 默认值 | `bmm1_scale=D^-0.5` |
| workspace | 无额外 Python workspace | 128 MiB，prefill/decode 共用 | 128 MiB |
| 内部选择 | SM100 选版本 4，否则版本 3 | wrapper 强制 `backend=fa2` | TRTLLM FMHA |

### FlashInfer 的 plan 有额外并发约束

FI 的 **prepare_metadata** 只创建 FIMetadata，真正的 wrapper.plan 延迟到 forward。plan 会复用 pinned host staging buffer，并发起异步 H2D；下一次 plan 前必须等待上一个 CUDA event，避免 host buffer 在拷贝结束前被改写。CUDA Graph 路径同样通过 plan 把动态值写进固定 buffer。这个同步点属于正确性约束，性能分析时不能随意删除。

实现还直接设置 FlashInfer wrapper 的 `_int_workspace_buffer` 和 `_backend` 私有字段。项目依赖只限定 `flashinfer-python >= 0.5.3`，升级依赖时应把这两处纳入兼容性回归。

### 卡内 split-KV 增加并行任务

当 Decode 或短 chunk 续算遇到极长历史 KV 时，若只按 Query 行和 Head 切分线程块（CTA），单张 GPU 上的并行块数往往填不满全部 SM。FlashInfer 等推理后端为此引入**卡内 split-KV**：调度器在 `plan` 阶段把同一个 Q tile 对应的长 KV 历史切成多段交给不同 CTA 并行扫描，再用第 2 章推导的 Online Softmax 对数指数和（`LSE`）重缩放公式合并局部输出。它与第 15 章将要展开的跨 GPU 上下文并行（CP）使用相同的数学归约关系，但资源边界截然不同。

#### 分块加载和并行拆分有不同含义

Attention score 矩阵可写成 $S=QK^\top\in\mathbb{R}^{N_q\times N_{kv}}$。普通分块 attention 通常先沿 Q 的行维度切 tile，让不同 CTA 处理不同 Q tile；每个 CTA 再沿 KV 维度逐块扫描，并用 online softmax 累积输出。同一个 CTA 顺序加载 KV 块，也属于分块计算，但没有增加 KV 方向的并行任务。

| 执行方式 | 同一个 Q tile 对应的 KV 工作 | 并行范围 |
|-|-|-|
| 普通分块 attention | 一个 CTA 逐块扫描允许访问的 KV | 本卡主要沿 Q tiles 和 heads 分工 |
| 卡内 split-KV | 多个 CTA 各扫描一段 KV，再合并局部结果 | 同一张 GPU 内增加 KV 方向的任务 |
| 跨卡 CP | 多张 GPU 分摊上下文，跨卡通信满足远端 attention 依赖 | 多个 GPU 协作，卡内仍可采用各自的 kernel 调度 |

例如，一个 Q tile 需要关注 128K 历史 KV，split-KV 可以将历史分成四段，让四个 CTA 各处理一段，再合并局部输出和 LSE。合并沿用前文“局部 softmax 输出必须加权合并”的数学关系，不能直接求和或平均。卡内合并通过本卡 workspace、内存访问和同步完成；具体实现也可以融合计算与合并，避免额外阶段的启动开销。

Prefill CP 的通信路径则取决于实现。Ring 路径在 Q 所属 GPU 上逐块接收远端 K/V、计算并累积结果，因此不能一概描述为“每卡计算同一个 Q 的局部输出，再跨卡 merge”。后者更接近 KV 分片的 Decode CP。CP 的各类路径见前文 Prefill 和 Decode 小节。

#### split-KV 改善利用率，不扩充显存

| 比较项 | 卡内 split-KV | 跨卡 CP |
|-|-|-|
| 主要目标 | 补足 CTA 数量，减轻变长任务的负载不均衡 | 利用更多 GPU 的计算、HBM 带宽和可用容量 |
| 持久 KV Cache | 仍驻留在同一张 GPU，原 KV payload 不会按 split 数缩小 | 确实分片且无复制时，每卡只驻留部分 KV；部分 Prefill CP 会复制完整 KV |
| 带宽收益 | 提高已有 HBM 带宽的利用率，硬件总带宽上限不变 | 可使用多张 GPU 的 HBM 带宽，同时受跨卡互联限制 |
| 新增代价 | 局部输出和 LSE 的存储、读写与 merge，调度及部分重复的 Q 加载 | 远端 K/V 或 Q／局部结果传输，跨卡同步，以及布局转换 |
| 优化边界 | 主要优化本卡 attention kernel，不自动分摊投影和 FFN | Prefill 按 token 分工时也可分摊本地投影和 FFN；Decode 需检查复制与 TP 配合 |

当只沿 Q 和 heads 划分得到的任务不足时，长 KV 被拆成更多任务，能让更多 SM 同时工作。FlashInfer 还考虑不同请求的长度，将 query tile 对应的 KV 工作分段并分配给 CTA，减少部分 SM 已空闲、其他 SM 仍在处理长任务的情况。中间结果和 merge 都有成本，split 数增加不会保证单调变快。

#### 长 full prefill 通常已有很多 Q tiles

Full prefill 中，通常有 $N_q\approx N_{kv}\approx N$。即使不做 split-KV，也能沿 Q 产生很多任务。以 128K 个 Q、每个 Q tile 处理 128 行为简化例子，仅 Q 维就有：

$$131072/128=1024$$

个 tile，还未计入 batch 和 heads。实际 CTA 映射受 GQA、tile 配置和 backend 影响；causal attention 中，前部 Q 访问的 KV 更少，不能把每个 tile 都按完整 128K 历史估算。这个算例只说明长 full prefill 往往已有较多 Q 方向任务，不证明某个 kernel 已充分利用 GPU。

| 本轮 attention 形状 | 卡内 split-KV 的可能收益 | CP 的主要收益 |
|-|-|-|
| 长 full prefill，Q 和 KV 都长 | 改善局部调度和负载均衡；已有足够 CTA 时，收益可能较小 | 分摊大量计算，分片布局还可缓解每卡显存压力 |
| 短 chunk、长历史的 prefill／extend | Q tiles 少，每个 tile 的 KV 工作长，增加 KV 方向任务更有意义 | 分摊历史 KV 和计算，具体取决于布局与通信 |
| 普通 Decode，每请求一个 Q | 在本卡增加历史 KV 扫描任务，改善利用率 | 分摊长历史 KV 的容量和读取 |

例如，一条请求已有 128K 历史，本轮只处理 128 个新 token。它属于 prefill／extend，但 Q 很短，形状更接近 split-KV 容易获益的场景。判断时应看本轮 $N_q$ 和 $N_{kv}$，不能只依据“总 prompt 很长”。长 full prefill 中，如果本卡已接近计算或 HBM 带宽上限，更多 CTA 和 merge 可能只增加开销。

#### CP 与卡内拆分可以组合

可以先用 CP 把请求分到多张卡，再让每张卡的 FlashInfer 对本地 attention 工作做 split-KV。但二者会改变彼此的收益条件：CP 分掉部分 Q 后，每卡 Q tiles 变少，卡内拆分可能重新有用；本卡任务仍充足时，继续增加 splits 会增加中间结果和 merge；主要瓶颈已是跨卡通信时，卡内拆分难以消除这部分等待。两种优化的加速比不能直接相乘。

**CP 的判断：**新增 GPU 资源带来的收益，能否覆盖跨卡通信与同步。**卡内 split-KV 的判断：**更好的 SM 利用率和任务均衡，能否覆盖中间结果与 merge 成本。实验应固定请求分布、KV 布局、backend 和 GPU 资源，分别记录本卡利用率、attention 及 merge 耗时、跨卡通信、TTFT／TPOT 和峰值显存。

机制依据为 [FlashInfer 论文 §3.3 的负载均衡调度与 attention 状态合并](https://arxiv.org/html/2501.01005v2)。论文描述 query tiles 的 KV 工作分段、任务分配、局部输出及归一化统计量，并区分 plan 和 run；具体 backend 不一定启用同一拆分策略。参数与支持范围应以 [FlashInfer 官方 attention 文档](https://docs.flashinfer.ai/api/attention.html)和实际安装版本为准。Mini-SGLang 的 wrapper.plan 生命周期与并发约束见“FlashInfer 的 plan 有额外并发约束”，不能仅凭论文机制推断当前部署的 split 行为或性能。

## GQA、MQA 不需要按 Q head 展开 K/V

ModelConfig 把全局 Q head 数 Gq 取自 **num_attention_heads**，全局 KV head 数 Gkv 优先取 **num_key_value_heads**，缺失时退化为 MHA。AttentionLayer 要求 Gq 能被 Gkv 整除。三个 kernel 直接接收 rank-local Hkv，运行时没有按 Q head 执行 `repeat_interleave`；当 TP 大于 Gkv 时，权重加载会把同一个 KV head 复制到多个 rank。

| 场景 | head 规则 | 工程含义 |
|-|-|-|
| MHA | `Gq == Gkv` | 每个 Q head 有独立 K/V head。 |
| GQA | `1 < Gkv < Gq` | 多个 Q head 共享一个 KV head；FI 在全局分组比不小于 4 时默认启用 tensor-core decode。 |
| MQA | `Gkv == 1` | 所有 Q head 共享一个 KV head，KV Cache 体积最小。 |
| `TP <= Gkv` | Q/KV heads 都按 TP 正常切分 | 每个 rank 保存自己的 KV shard。 |
| `TP > Gkv` | 要求 `TP % Gkv == 0`，每 rank 保留一个 KV head | 多个 rank 复制同一个 KV head；权重加载、projection、KV pool 和 FI plan 使用同一规则。 |

源码形状支持 MQA 与 KV-head replication，但仓库没有针对三 backend 的 GQA/MQA 数值测试。这里应表述为“实现路径存在且静态一致”，不能写成“所有组合已验证”。

## CUDA Graph 固定地址，replay 只更新动态值

`GraphRunner` 的捕获、bucket 补齐、静态缓冲与重放生命周期见下一章〈[CUDA Graph 与双流流水线隐藏 Decode 开销](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyX1IsQtSwPsMTTOoye90aqg)〉。本节只标出三个后端在重放前的 metadata 更新差异：FA 与 TRTLLM 原地覆写长度与有效块表列；FI 切换到对应 bucket 的 decode wrapper 并重新执行 `plan`，把 ragged 索引写进固定地址。

<callout emoji="💡">
**待验证边界：**多个 dummy request 会向同一个 dummy `out_loc` 槽位写入 K/V。设计上它们输入相同、写入值也相同，但现有 `test_store.py` 仅测试互不重复的目标索引，尚未覆盖 graph padding 下的并发同址写入。
</callout>

## 自动选择把 GPU 架构和页大小绑在一起

| 条件 | auto 结果 | 页大小影响 |
|-|-|-|
| SM100 及以上 | `trtllm` | 只接受 16、32、64；其他值会被改成 64。 |
| SM90 | `fa,fi` | FA prefill、FI decode，共用用户 page size。 |
| 更早架构 | `fi` | FI 内部始终用逻辑 page size 1 解释 cache。 |

CLI 的 `--attention-backend`／`--attn` 接收单 backend 或 `prefill,decode`。只要配置字符串含 `trtllm`，整个系统 page size 都受 TRTLLM 约束，混合配置也不例外。EngineConfig 虽是 frozen dataclass，**_adjust_config** 仍会通过 `object.__setattr__` 改写 backend 和 page size，排查“参数为何没有按输入生效”时应先看这里。

## 现有测试只覆盖了链路的一小段

仓库没有直接测试 FAMetadata、FIMetadata、TRTLLMMetadata、HybridBackend、RoPE、GQA/MQA 或三个 backend 的数值一致性。现有相关证据主要有两类：**test_cache_allocate.py** 检查 P_page 大于 1 时的 page 对齐、eviction 和 page-to-token 展开；**test_store.py** 在 CUDA 上比较唯一随机 slot 的 K/V 写入结果。**test_scheduler.py** 更接近依赖真实模型的手工端到端脚本，并没有 pytest 会收集的 test 函数。

**地址解释变化的优先验证面：**最小对照应同时覆盖无命中与部分命中、分块 Prefill 与 Decode、页大小切换、GQA/MQA 与 KV head 复制，以及 eager 与 graph padding。先手算 `positions`、`out_loc`、`cu_q/cu_k`，再与 Hugging Face eager 参考实现比对各层输出与耗时（同时统计 metadata 准备、H2D、`plan` 与 kernel 时间）。完整源码入口见〈[附录 A：Mini-SGLang 实战、排障与查阅手册](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyVjgMnxC8UnNtNPHuubayUe)〉。

## 证据边界与验证重点

现有单元测试仅覆盖了 CPU 页分配账本与唯一随机槽位的 K/V 写入，尚未直接验证三种后端、GQA/MQA、TP 复制、RoPE 及 CUDA Graph padding 的端到端数值等价。

<callout emoji="✅">
**本章最小验证：**给定两条不等长请求的进度与页表，手算 `positions`、`out_loc`、`cu_q` 与 `cu_k`；至少覆盖一次前缀命中、一次 MQA/TP 复制和一次 graph padding，对齐 FA、FI 与 TRTLLM 的输出。全系统实验矩阵见第 12 章〈[自定义 Kernel 与系统收官实验](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyCrT9cfNBzyzOQfBHC0SB5U)〉。
</callout>

Attention Backend 是调度地址计划、模型 Q/K 语义、KV Cache 物理布局与 GPU Kernel 契约汇合的位置。下一章继续沿已有执行链路，说明 CUDA Graph 怎样固定模型计算、更新动态元数据，并与 CPU 结果处理形成流水线。量化、约束解码、投机验证和新模型状态则在进阶专题中分别展开。

# CUDA Graph 与双流流水线隐藏 Decode 开销

```{contents} 本页目录
---
depth: 2
local: true
---
```

本章默认站在一个 TP rank 的进程内观察：该 rank 的 `Scheduler` 持有自己的 `Engine`，`Engine` 持有自己的 `GraphRunner` 和 engine stream，`Scheduler` 另有一条 scheduler stream。在线 TP > 1 时，各 rank 分别 capture 和 replay 本地图，但 `model.forward` 中的 TP collective 仍跨 rank 配对执行。下文从 forward 的数据形状出发，说明 Decode 模型计算、graph 的静态与动态边界，以及两条本地 CUDA stream 的 overlap 时序。

<callout emoji="📐">
**源码核对边界：**本章依据主干基线 `9a91cfa` 静态核对 capture/replay 与双流同步契约；数值等价、跨 TP rank 随机采样一致性及共享可变 `Req` 的边界表现，需按文末方案在目标 CUDA 机器上实测确认。
</callout>

## 先把 Batch 的三种长度分清

设 **B** 为真实请求数，**B_p** 为 graph bucket 补齐后的请求数，**T** 为本轮所有请求 extend_len 之和，**H** 为隐藏维度，**V** 为词表大小。Prefill 中 T 可以大于 B；decode 每请求贡献一个 token，命中 graph 时 B_p 可以大于 B。

### 关键张量的 shape

| 对象 | Prefill | Decode eager | Decode graph |
|-|-|-|-|
| 真实请求数 | B | B | B |
| padded_reqs | B_p＝B | B_p＝B | B_p＝首个不小于 B 的捕获尺寸 |
| input_ids、positions、out_loc | [T] | [B] | 静态 buffer 的 [B_p] slice |
| Transformer hidden | [T, H] | [B, H] | [B_p, H] |
| 模型返回 logits | [B, V] | [B, V] | 内部 [B_p, V]，返回时裁成 [B, V] |
| 采样 token | [B] | [B] | [B] |

### ForwardInput 和 ForwardOutput 各自保存什么

Scheduler._prepare_batch 产生 ForwardInput。模型从全局 Context 取 Batch；ForwardInput 是当轮 Batch 与读写映射的载体。映射在准备阶段固定，但其中的 Batch 和 Req 仍是共享可变对象：

- **batch**：Req、phase、positions、out_loc 和 attention metadata。
- **sample_args**：真实请求的 temperature、top-k、top-p GPU 张量。
- **input_tuple**：token_pool 的请求行与 token 位置，用于 gather input_ids。
- **write_tuple**：真实请求行与 next-token 位置；不能继续 decode 时位置为 -1。它必须在 Req 状态推进前保存。

ForwardOutput 中两份 token 都是 [B]。Engine 先生成 int32 GPU token，再排入 non_blocking D2H copy 并记录 copy_done_event；forward_batch 返回后，Scheduler._forward 才在同一 engine stream 把 GPU token 写回 token_pool。Scheduler 在读取 CPU token 前同步 copy_done_event。

## Engine.forward_batch 的语义很窄

### 一次调用按固定顺序完成六件事

1. 断言当前 CUDA stream 就是 Engine 自己的 stream。
2. 用 Context.forward_batch 安装当前 Batch；无参数的 BaseLLMModel.forward 从全局 Context 读取输入和 metadata。
3. 若 batch 是 decode，且 B 不大于 max_graph_bs，则调用 GraphRunner.replay；其余情况调用 model.forward 走 eager。
4. 只对真实请求调用 Req.complete_one：cached_len 取旧 device_len，device_len 加一。Dummy 不推进。
5. 只把 logits 的前 B 行交给 Sampler，确保 padding 行不进入采样。
6. 把采样结果异步复制到 CPU，记录 copy_done_event，然后同时返回 GPU、CPU 两份 token；token_pool 写回由调用方 Scheduler._forward 接着排入同一 stream。

```python
assert current_stream == engine_stream
with ctx.forward_batch(batch):
    if batch.is_decode and batch.size <= max_graph_bs:
        logits = graph_runner.replay(batch)
    else:
        logits = model.forward()

for req in batch.reqs:
    req.complete_one()

next_gpu = sampler.sample(logits[:batch.size], sample_args).to(int32)
next_cpu = next_gpu.to("cpu", non_blocking=True)
copy_done.record(engine_stream)
```

无参数入口让 eager 和 graph 共用模型代码。进程级 Context 同时只能挂一个 active batch；动态信息要提前写入 Batch，或在 replay 前复制到静态 buffer。

### Eager 和 graph 的分界只看 phase 与真实请求数

| 条件 | 执行路径 | 原因 |
|-|-|-|
| Prefill | 始终 eager | T 随 prompt 和 prefix hit 改变，单批 shape 不固定 |
| Decode，B 在 graph 上限内 | 补到 bucket 后 replay | 每请求固定一个 query，可以把动态 batch 映射到少量静态 shape |
| Decode，B 超过 graph 上限 | eager | 没有可容纳该批次的已捕获图 |
| 未显式设置 cuda_graph_bs，且 cuda_graph_max_bs 小于 1 | 全部 eager | GraphRunner.max_graph_bs 为 0，capture 直接返回 |

从单个 rank 看，Graph 只包住该 rank 的 `model.forward` 和固定 logits buffer 写入；TP > 1 时，`model.forward` 内的 all-reduce 和 all-gather 也位于 capture 区间。捕获成功后，这些 collective 随各 rank 的本地图 replay，并在 rank 间配对执行。KV page 分配、metadata 构造、采样和 D2H copy 都在图外，端到端收益不能全部记到 replay 名下。

## Capture 固定执行拓扑，replay 更新数据

### Capture 在 Engine 初始化时完成

每个 TP rank 的 `Engine` 都在初始化时创建自己的 `GraphRunner`，并把本 rank 的 engine stream、模型分片和 attention backend 传给它。`GraphRunner` 在这条 engine stream 上完成 warmup 和 capture，同时计算捕获尺寸、准备 Decode backend workspace，并分配本 rank 长期存活的 `GraphCaptureBuffer`：

- input_ids、out_loc、positions 容量为 max_graph_bs；logits 为 [max_graph_bs, V] float32。
- 每个 bucket 用 B_p 个 dummy Req 构造 decode Batch，先 eager warmup，再捕获相同的 model.forward。
- Bucket 从大到小捕获，后续图复用第一张图的 memory pool。

图引用的权重、workspace、输入和 metadata buffer 地址必须稳定。模型、dtype、TP 拓扑、backend、page size 或最大序列长度变化时都要重建 Engine。

### Replay 前只改允许变化的值

每个 rank 的 `GraphRunner` 先把 padded Batch 的 input_ids、out_loc 和 positions 复制进本地静态 buffer，再按 B_p 选图。Backend 更新长度、page table 或 graph wrapper plan，随后 `Engine.forward_batch` 在本 rank 的 engine stream 上调用 `g.replay`，并只返回前 B 行 logits。TP > 1 时，各 rank 必须命中同一个 B_p，图内 collective 才能按相同顺序配对。

| 边界 | 固定内容 | 每轮可变内容 |
|-|-|-|
| 模型 | 权重地址、算子序列、dtype、TP collective 拓扑 | 输入 token 对应的数据值 |
| Batch | bucket B_p、各 capture tensor 的地址和 shape | 真实 B、token id、position、KV 写入位置 |
| Attention | workspace、capture metadata buffer 地址、decode kernel 路径 | seq_lens、cu_seqlens、page table 或 FlashInfer plan |
| 图外 | 采样算法和 Scheduler 控制流不属于 graph | sampling 参数、随机状态、D2H 回传、请求结束判断 |

Decode 的 kernel 拓扑稳定，replay 能把一串 CPU kernel launch 合成一次图提交。它不会减少矩阵计算；B_p－B 个 dummy 行仍会执行。Bucket 越稀，padding 越多；越密，常驻图和 capture 成本越高。

### 用收益上限判断 Graph 是否值得

假设一轮 eager Decode 的 CPU 提交与调度占 20%，GPU 计算及其他不可省工作占 80%。即使提交开销全部消失，理想加速上限也只有 $1/0.8=1.25$ 倍；实际还存在 graph launch、plan、静态 buffer copy 和 dummy 行成本。若主要瓶颈已经是长上下文的 KV 读取，Graph 收益可能很小。应先从 timeline 量出可省的那部分，再判断是否值得增加捕获尺寸。

## Bucket 和 dummy 让动态 batch 落到静态 shape

### Bucket 的默认选择与配置陷阱

未指定 cuda_graph_bs 时，初始空闲显存超过 80 GiB 默认上限 256，其余为 160；尺寸是 1、2、4，以及从 8 起每隔 8 一个。B＝3 补到 4，B＝5 补到 8；超过最大尺寸则 eager。

<callout emoji="💡">
**配置陷阱：**未显式设置 `cuda_graph_bs` 时，默认列表固定先放入 `1, 2, 4`，因此将 `max` 设为 1、2 或 3 仍会捕获到 4（设为 `0` 才彻底关闭）；显式传入 `cuda_graph_bs` 时则会直接忽略 `cuda_graph_max_bs`。默认上限取自模型加载前的初始空闲显存，捕获期间 OOM 不会自动降档。
</callout>

### Dummy 只占静态槽位，不进入请求语义

Dummy Req 使用 table_idx＝max_running_req，对应额外 page table 行和额外 KV page，token_pool 初值为零。Padding 会在真实 reqs 后重复引用它，使图内 shape 达到 B_p。Engine 只推进 batch.reqs，只采 logits[:B]，write_tuple 也只含真实请求，因此 dummy 计算不会进入用户输出。

## Attention backend 决定 graph 能否安全 replay

各 backend 的 metadata 布局、页表解释和 FlashInfer plan 约束见 Attention 章节；这里仅说明它们怎样参与 graph capture 与 replay。

### 三类 backend 的共同契约

BaseAttnBackend 约定 graph 初始化、capture 准备和 replay 准备。HybridBackend 只把这些调用转给 decode backend。

| Backend | Capture 方式 | Replay 前更新 | 约束 |
|-|-|-|-|
| FlashAttention | 固定长度与二维 page_table buffer | 复制长度和有效列 | 实际长度屏蔽表尾 |
| TensorRT-LLM | 固定 metadata buffer | 复制长度和 block table | page size 限 16、32、64；否则改为 64 |
| FlashInfer | 每 bucket 一个 graph decode wrapper | 切 wrapper 并 plan | 接口使用逻辑 page_size＝1 |

FlashInfer `plan` 复用 pinned host staging buffer 并发起异步 H2D；`_initialize_metadata_once` 在下次复用前仅同步上一次 `last_event` 以消除 metadata 竞态，不会触发整卡级别的全局同步。

### TP 模式还要求所有 rank 同步命中同一条路径

在线 TP 模式为每个 rank 启动一个 `Scheduler` 进程；每个进程各自创建 `Engine`、`GraphRunner`、模型分片和本地 CUDA Graph。“每个 rank 一张本地图”不表示图内没有通信：TP > 1 时，`model.forward` 中的 all-reduce 和 all-gather 仍位于 capture 区间；捕获成功后，它们成为各 rank 本地图中的通信节点，replay 时跨 rank 配对执行。所有 rank 必须使用相同请求顺序、B 和 bucket；还要在目标 NCCL 或 PyNCCL 版本验证 graph capture，单卡通过不能代表多卡通过。

## 采样与回传留在图外

当前 Graph capture 只包住 `model.forward`；Sampler 参数准备、随机抽样与异步 D2H 回传均留在图外执行。完整采样分支、混合批低温近似与 TP 随机状态要求见第 9 章〈[模型执行与采样：从 Batch 到 token](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyTjwgnLCFNIX7iKHS0V2TVh)〉。

## Overlap 把 CPU 工作挪到下一批计算后面

请求状态、资源交接和 abort 边界见 Scheduler 章节；这里仅保留两条 stream 与 token 回写的执行时序。

### 两条 stream 怎样接力

以下时序在每个 TP rank 内各有一份：本 rank 的 scheduler stream 准备 page table、positions、metadata 和采样张量，engine stream 先等待 scheduler stream，再执行模型与采样。`wait_stream` 只约束本 rank 的两条 stream；跨 rank 通信依赖由 `model.forward` 内的 collective 提供：

```text
scheduler stream: 收消息 → 选当前批 → 准备 metadata 和映射
engine stream:    wait_stream(scheduler) → 当前批 forward，排入 D2H 并记录 event → GPU token 写回 token_pool
CPU:              同步上一批 copy_done → 追加 host token → 判断 EOS/完成 → 回收资源 → 回复
下一轮输入依赖 GPU token_pool，不必等待上一批 D2H 和 detokenize
```

ForwardInput 固定当轮读写映射，但其中的 Batch 和 Req 仍是共享可变对象。Engine.forward_batch 返回后，Scheduler._forward 在同一 engine stream 排入 GPU token 写回；CPU token 只用于输出、EOS 和 host 历史。

### 共享可变 Req 带来一个待验证边界

<callout emoji="💡">
**静态推导风险：**Overlap 会先在 GPU 流启动当前批（同步执行 `Req.complete_one()`），随后才在 CPU 上处理上一批输出；当上一批检查 `req.can_decode` 判断输出额度是否耗尽时，读取到的可能是已被当前批提前加一的 `device_len`，EOS 判定前也可能已在途发出一轮额外 decode。
</callout>

finished_reqs 只防重复释放，不撤销在途计算。在线前端首个 finished 后停止等待，离线 LLM 的收集逻辑不同。需用 max_tokens＝1、2、3 和首 token EOS 的可控 stub，对比 overlap 开关下的 token 数、finished 时机、释放次数和晚到消息；实验前不能断言一定会多返回 token。

## 用 Nsight 把 graph 收益和流水线收益拆开

完整消融矩阵与指标口径见实践章节；这里聚焦 CUDA Graph 和两条 stream 的观测点。

### 实验矩阵先固定变量

固定 GPU、模型、backend、TP 和 prompt。Graph off 使用 --cuda-graph-max-bs 0，graph on 选择覆盖测试 batch 的上限。两组都先 warmup，capture 时间与稳态 latency 分开记录；overlap 另加 MINISGL_DISABLE_OVERLAP_SCHEDULING=true 对照。

```bash
nsys profile --trace=cuda,nvtx,osrt -o graph_off \
  ${SERVER_CMD} --cuda-graph-max-bs 0

nsys profile --trace=cuda,nvtx,osrt -o graph_on \
  ${SERVER_CMD} --cuda-graph-max-bs 8
```

${SERVER_CMD} 使用目标机已验证的启动命令。Batch size 覆盖 1、2、3、4、5、7、8、max_graph_bs 和 max_graph_bs＋1，观察 padding 与 eager fallback。

### Trace 中要看这些证据

| 证据 | Graph on 的预期 | 异常提示 |
|-|-|-|
| CPU CUDA API launch | 模型段由多次 kernel launch 变成 graph launch，CPU 提交时间下降 | 仍逐 kernel launch，说明未命中 replay 或 graph 被打断 |
| GPU kernel 主体 | 真实计算量相近；P 大于 B 时出现可解释的 padding 开销 | 同 bucket kernel 拓扑变化，需检查动态控制流或 backend plan |
| Scheduler 与 Engine stream | metadata/H2D 与上一批 GPU 计算重叠，engine stream 只在依赖点等待 | 频繁 device synchronize 或大段 GPU 空洞，CPU 工作没有被隐藏 |
| Sampler 和 memcpy DtoH | 仍在 graph 外，可与下一轮准备形成流水 | 把它们误算成 graph 内收益，或 copy_done 过早同步 |
| FlashInfer plan | 每批可见 planning 与 event 依赖，但无 staging buffer 竞态 | plan 串行占比过高，graph 只省下 launch，端到端收益有限 |

报告至少给出 prefill latency、decode 每 token latency、端到端 p50/p99、padding 比例、CPU launch 时间和 graph 显存。只报 tokens/s 无法区分 replay、batch 与 overlap 收益。

## 证据边界与验证重点

CUDA Graph 的 capture/replay 边界、bucket 与静态 buffer 可以由源码确认；graph on/off 的数值等价、不同 backend 的 replay、TP 多 rank collective、dummy slot 写入与 overlap 完成时机仍需在目标 GPU 上验证。

<callout emoji="✅">
**本章最小验证：**覆盖 bucket 精确命中、间隙 padding 与超上限 eager 回退三类批次，确认真实请求的 token、KV 与长度推进不受 dummy 行污染；再用 Nsight Systems 拆开 graph launch、backend plan、sampling、D2H 与双流重叠占比。全系统消融要求见下一章〈[自定义 Kernel 与系统收官实验](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyCrT9cfNBzyzOQfBHC0SB5U)〉。
</callout>

CUDA Graph 减少模型段的重复提交，Overlap 则把部分 CPU 工作与 GPU 执行重叠。TP 的同序执行要求已经在前文建立，Graph replay 仍须遵守它。下一章继续追踪这些执行路径调用的 C++、CUDA 与 Triton Kernel，检查输入契约和内存访问。

# 附录 A：Mini-SGLang 实战、排障与查阅手册

```{contents} 本页目录
---
depth: 2
local: true
---
```

本附录将安装启动、接口兼容子集、常见故障与静态待验证边界、分阶段学习路线、完整源码索引、术语表及 CLI 参数合为一站式速查手册（上游核对基线为 `9a91cfa`，延迟准入参数核对至本地扩展 `467f2eb`）。

## 先把 Mini-SGLang 跑起来

<callout emoji="💡">
运行需要 Linux（x86_64 或 aarch64）、NVIDIA GPU，以及和驱动版本匹配的 CUDA Toolkit（部分 kernel 在运行时 JIT 编译）。macOS 和 Windows 不能直接运行，Windows 可以用 WSL2 或 Docker。本章命令摘自仓库的 README 和 `docs/features.md`，本文写作时没有在 GPU 上执行过。
</callout>

### 安装

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

```

依赖在 `pyproject.toml` 里：`torch<2.10.0`、`transformers>=4.56.0,<=4.57.3`、`flashinfer-python>=0.5.3`、`sgl_kernel>=0.3.17.post1`、`apache-tvm-ffi>=0.1.4`、`pyzmq`、`fastapi`、`uvicorn` 等。也可以用仓库自带的 Dockerfile：

```bash
docker build -t minisgl .
docker run --gpus all -p 1919:1919 minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
# 交互式 shell
docker run -it --gpus all minisgl --model Qwen/Qwen3-0.6B --shell-mode
```

第一次启动会比较慢：要下载权重、JIT 编译 TVM FFI kernel 和 FlashInfer kernel、捕获几十张 CUDA graph。README 建议给 `huggingface`、`tvm-ffi`、`flashinfer` 三个缓存目录挂 Docker volume。

### 启动在线服务

```bash
# 单卡部署 Qwen3-0.6B，默认监听 127.0.0.1:1919
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 4 卡张量并行，端口改为 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp-size 4 --port 30000

# 从 ModelScope 下载权重
python -m minisgl --model "Qwen/Qwen3-32B" --tp-size 4 --model-source modelscope
```

注意 `torch.distributed` 的初始化地址是 `tcp://127.0.0.1:{port+1}`（`ServerArgs.distributed_addr`），所以端口号加 1 也必须空闲。同一台机器上起多个实例时，ZMQ 的 IPC 地址带了主进程 pid 后缀，不会冲突。

### 交互式 shell

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell-mode
# 等价于：python -m minisgl.shell --model "Qwen/Qwen3-0.6B"
```

`--shell-mode` 是规范参数名；`--shell` 只是 argparse 当前允许的无歧义前缀。该模式把 `cuda_graph_max_bs` 与 `max_running_req` 配置值设为 1；GraphRunner 的默认 bucket 规则仍可能捕获到 4。Scheduler 进入主循环前会屏蔽 INFO 及更低级别日志，初始化与前端仍可能输出日志。输入 `/reset` 清空本地对话历史，`/exit` 或 Ctrl-D 退出。采样由 MINISGL_SHELL_MAX_TOKENS、MINISGL_SHELL_TEMPERATURE、MINISGL_SHELL_TOP_K 和 MINISGL_SHELL_TOP_P 控制，四项默认值依次为 2048、0.6、-1 和 1.0。

### 调用 OpenAI 兼容接口

```bash
curl -N http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "用一句话解释 KV cache"}],
       "max_tokens": 128, "stream": true}'

```

把 `"stream"` 改成 `false` 会一次性返回完整的 `chat.completion` JSON（PR #125 加入）。用官方 Python SDK 也可以：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="dummy")  # api_key 随便填，服务端不校验
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

```

除 `/v1/chat/completions` 外，`server/api_server.py` 还提供接收纯文本并以 greedy 默认参数流式输出的 `/generate`，以及返回当前模型路径的 `/v1/models`。

### OpenAI 兼容接口的源码边界

“OpenAI-compatible” 表示现有客户端可以走熟悉的路径和基础字段，不表示协议的每一项语义都已实现。阅读或接入时，应以当前源码为准。

| 边界 | 当前源码行为 | 接入影响 |
|-|-|-|
| 路由 | 实现 `/v1/chat/completions`，没有 `/v1/completions` 路由。 | 只按 chat completions 接入。 |
| `model` | 不参与模型路由或校验；非流式响应会原样回显。 | 不能靠该字段在一个服务内切模型。 |
| 声明但未下传的字段 | `n`、`stop`、`presence_penalty`、`frequency_penalty` 没有进入 `SamplingParams`。 | 不要把这些参数写进正确性假设。 |
| 停止条件 | 由最大输出长度和 EOS 决定，没有 stop-string 检测。 | `stop` 字符串不会截断输出。 |
| usage | 非流式响应中的三项 token 计数固定为 0。 | 不能直接用于计费或容量分析。 |
| 流式 schema | chunk 的 `object` 是 `text_completion.chunk`，id 前缀是 `cmpl-`；结束原因统一写 `stop`。 | 严格校验标准 chat schema 的客户端可能需要适配。 |
| 认证 | 当前 FastAPI 路由未看到 API key 校验；仓库 benchmark 使用 dummy key。 | 生产暴露前需要由部署层补齐访问控制。 |
| 超长输入 | scheduler 记录 warning 后直接丢弃，未向前端发送终止回复。 | 客户端可能持续等待；这是需要补测和修正的失败边界。 |

### 离线推理与压测

```python
from minisgl.core import SamplingParams
from minisgl.llm import LLM

llm = LLM("Qwen/Qwen3-0.6B")   # 其余关键字参数透传给 SchedulerConfig，如 page_size、max_extend_tokens
outs = llm.generate(["介绍一下 radix tree"], SamplingParams(temperature=0.6, max_tokens=128))
print(outs[0]["text"], outs[0]["token_ids"][:8])

```

```bash
# 离线吞吐：256 条随机长度请求（输入、输出各 100~1024 token），输出 tok/s
python benchmark/offline/bench.py
# 关闭 overlap scheduling 做消融对比
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python benchmark/offline/bench.py
# 用 WildChat 真实 prompt 的离线测试（会下载 parquet 分片）
python benchmark/offline/bench_wildchat.py

# 在线压测：先在 1919 端口启动服务，再运行
python benchmark/online/bench_simple.py   # 64 条随机请求并发
python benchmark/online/bench_qwen.py     # 回放 Qwen trace 前 1000 条，按不同时间缩放比例各跑一轮

```

调试时可以设置 `LOG_LEVEL=DEBUG` 打开 debug 日志，`LOG_PID=1` 在日志前缀里加上进程号。

### 常用参数速查

环境变量集中定义在 `env.py` 的 `EnvClassSingleton`，统一加 `MINISGL_` 前缀：`MINISGL_DISABLE_OVERLAP_SCHEDULING`（关闭 overlap）、`MINISGL_FLASHINFER_USE_TENSOR_CORES`（强制 FlashInfer decode 是否用 tensor core）、`MINISGL_PYNCCL_MAX_BUFFER_SIZE`（PyNCCL 对称内存缓冲上限，默认 1 GiB，支持 `512M` 这样的写法），以及上面提到的四个 `MINISGL_SHELL_*`。

## 常见问题与易错点

### 运行时常见问题

| 现象 | 先查什么 | 进一步判断 |
|-|-|-|
| 启动报显存不均衡 | 逐卡可用显存和其他进程 | 差异超过 2 GiB 会被同步检查拒绝；阈值内也不代表各卡容量都安全 |
| 启动时报端口占用 | 服务端口与 port+1 | 后者用于 distributed 初始化 |
| 回复很短或每次不同 | max_tokens 与 temperature | chat 默认最大输出 16、temperature=1.0；与 /generate 默认值不同 |
| stop、n、penalty 无效 | 接口字段到 SamplingParams 的映射 | 当前字段声明不代表后端已实现 |
| 请求迟迟没有终包 | 先按 UID 查 Tokenizer、Scheduler 与返回链 | 输入达到 max_seq_len 时源码只 warning 后丢弃是一个具体分支；也可能是前缀准入阻塞、GPU 停滞或 worker 退出 |
| 指定 TRTLLM 后页大小改变 | 最终 backend、page size 与 GPU 架构 | TRTLLM 只支持 16、32、64，其他值会改为 64 |
| 首启特别慢 | 权重下载、JIT、Graph capture 各阶段 | 分别保留阶段耗时；缓存持久化影响再次启动 |
| macOS 无法运行 CUDA 依赖 | 平台与工具链 | 本仓库 GPU 路径需要 Linux/NVIDIA 环境 |

### 读代码时的易错点

- `page_table` 里存的是逐 token 的 KV 位置，不是页号；page_size 大于 1 时由 FA 和 TRT-LLM 后端自己换算。
- `device_len` 包含本步要算的 token，`cached_len` 只包含 KV 已就绪的 token；forward 之后 `device_len` 还会再加 1。
- `ChunkedReq` 永远不会进入 decode 集合，它的采样结果也不会发给 detokenizer。
- `match_prefix` 会分裂节点，树的结构在匹配时就可能变化。
- `Req` 是 `eq=False` 的 dataclass，哈希按对象身份计算；凡是跨 TP rank 需要一致的顺序，都不能依赖 set 或 dict 的遍历顺序。
- `SamplingParams` 的默认值（`temperature=0`、`max_tokens=1024`）和 OpenAI 接口的默认值（`temperature=1.0`、`max_tokens=16`）不同。
- overlap 下 CPU 看到的请求状态比 GPU 晚一步，分析时序问题时先分清「CPU 簿记的值」和「GPU 上正在执行的 batch」。

### 高优先级待验证问题

| 静态发现 | 可能表现 | 最小验证 |
|-|-|-|
| Overlap 中下一批先推进共享 `Req.device_len` | 长度终止提前、EOS 后仍有在途回复 | `max_tokens=1/2/3`，对比 overlap 开关的回包数和 finished 次数 |
| 新请求 admission 按 token 估算，物理分配按页取整 | 多个短请求通过准入后仍可能无页可分配 | 小总页数、`page_size=4/16`、并发短请求 |
| 超长输入只记录 warning 后丢弃 | 前端可能一直等待终态 | 发送超过 max sequence 的请求并设置客户端超时 |
| TP 各 rank 独立采样，没有 token 广播 | 随机状态分歧可导致后续 collective hang | greedy 与随机采样分别记录每 rank UID 和 token |
| CUDA Graph dummy 请求共享写入槽位 | 多个 warp 对同一地址并发写 | 覆盖非 bucket batch size，对比 graph on/off 的真实请求输出与 KV |
| PyNCCL dtype map 只登记 FP16/BF16 | TP + FP32 可能在 dtype lookup 失败 | FP32 下比较默认与 `--disable-pynccl` |
| 词表不能被 TP 整除时，最后 rank 权重 shape 可能不匹配 | 权重加载 assert | 构造不可整除 tiny vocab checkpoint |

这些条目是源码推导出的测试优先级，不应直接写成已经复现的缺陷。工程判断要把“静态可疑”“可稳定复现”“根因已确认”分成三个证据等级。

## 分阶段学习路线

| 阶段与目标 | 必读源码 | 静态／CPU 主线 | Linux／NVIDIA GPU 可选线 | 验收问题 |
|-|-|-|-|-|
| 建立地图：认清进程、消息和 GPU rank。 | `README.md`、`docs/structures.md`、`server/launch.py`、`server/args.py`、`core.py` | 画进程表，标出 ZMQ、Gloo、NCCL 传什么；记录 SHA。 | 运行 `python -m minisgl --help`，验证参数入口。 | `Batch` 为何不跨 ZMQ？TP rank 为何各有 Scheduler 与 Engine？ |
| 追完 R0：从 HTTP JSON 到 SSE。 | `server/api_server.py`、`tokenizer/*`、`message/*`、`scheduler/io.py` | 记录 `TokenizeMsg → UserMsg → Req → DetokenizeMsg → UserReply`；观察 CPU Tensor 序列化。 | 请求流式、非流式、EOS、超长输入和取消，记录首包与终包。 | chat template 在哪应用？谁维护增量文本？取消能否抢占 kernel？ |
| 掌握调度与缓存：说清状态和所有权。 | `scheduler/{scheduler,prefill,decode,table,cache}.py`、`kvcache/{mha_pool,radix_cache}.py` | 手算共享前缀请求的三个长度、页数和 table row；运行 `tests/core/test_cache_allocate.py`。 | 比较 `radix／naive` 的冷、热前缀 TTFT；改变 prefill budget，记录显存与抖动。 | page、请求 slot、物理 token slot 有何区别？谁 lock、insert、evict、free？ |
| 读懂模型与 Attention：把 Batch 变成 logits。 | `models/*`、`layers/*`、`attention/{base,fa,fi,trtllm}.py` | 给 prefill／decode 写 shape 表；手算 QKV、gate/up 的融合与 TP 切分。 | 固定 greedy 输入，对比 `fa`、`fi`、`trtllm`、`fa,fi` 的逐 token 输出。 | backend 五个方法各负责什么？GQA 为何让 Q head 与 KV head 数不同？ |
| 理解执行与并行：掌握 graph、采样和 collective。 | `engine/{engine,graph,sample}.py`、`distributed/*`、`layers/{linear,embedding}.py` | 列出 Batch 字段生产者／消费者；手算 TP 与 collective shape。 | 比较 graph on／off 的正确性与 TPOT；TP＝1／2／4 对照两种 NCCL 路径。 | graph 为何只覆盖 decode？各 rank 须保持哪些不变量？ |
| 下钻 Kernel：把“快”拆成可验证假设。 | `kernel/{utils,index,store}.py`、`kernel/csrc`、`kernel/triton/fused_moe.py`、`benchmark` | 检查 shape、stride、dtype、index、stream 契约；先写朴素 PyTorch 参考实现。 | 跑正确性测试、compute-sanitizer、Nsight；分开首次 JIT、warmup 与稳定态。 | 瓶颈来自算力、带宽、launch、metadata、同步还是网络？怎样一次只改一个变量？ |

## 完整源码索引

| 问题域 | 源码 | 关键职责／符号 |
|-|-|-|
| 启动与 HTTP | `python/minisgl/__main__.py`；`server/args.py`、`launch.py`、`api_server.py` | `parse_args`、进程创建、`FrontendManager`、`v1_completions`、SSE、取消。 |
| Tokenizer | `tokenizer/server.py`、`tokenize.py`、`detokenize.py` | `tokenize_worker` 分流消息；应用 chat template；按 UID 维护增量解码。 |
| 消息协议 | `message/backend.py`、`frontend.py`、`tokenizer.py`、`utils.py`、`__init__.py` | `UserMsg`、`DetokenizeMsg`、`UserReply`、abort／batch；一维 CPU Tensor 编解码。 |
| 核心对象 | `core.py` | `SamplingParams`、`Req`、`Batch`、`Context`；长度、phase 与字段所有权。 |
| 调度总线 | `scheduler/scheduler.py`、`io.py`、`config.py`、`utils.py` | normal／overlap loop、prefill-first 选批、Batch 准备、结果处理、rank 0 广播与回复。 |
| Prefill／Decode | `scheduler/prefill.py`、`decode.py` | `PrefillAdder` 做命中、准入与 chunk；`DecodeManager` 管运行集合和在途预留。 |
| 请求表与页 | `scheduler/table.py`、`cache.py` | 请求 slot、token pool、page table；页分配、handle 交接、延迟释放、驱逐与账本检查。 |
| KV 与前缀 | `kvcache/base.py`、`mha_pool.py`、`radix_cache.py`、`naive_cache.py`、`__init__.py` | 物理 K／V 布局、`store_kv`、Radix 匹配／分裂／引用／驱逐；当前只创建 MHA pool。 |
| 模型装配 | `models/config.py`、`register.py`、`weight.py`、`base.py`、`utils.py` | HF 配置归一化、架构注册、safetensors 流式加载、TP 切片、QKV／gate-up 融合。 |
| 具体模型 | `models/llama.py`、`mistral.py`、`qwen2.py`、`qwen3.py`、`qwen3_moe.py` | 从一个 DecoderLayer 读通，再比较 bias、Q/K norm、dense／MoE 与 Mistral 映射。 |
| 通用层 | `layers/base.py`、`linear.py`、`embedding.py`、`attention.py`、`rotary.py`、`norm.py`、`activation.py`、`moe.py` | state dict、TP 线性与词表、RoPE、RMSNorm、激活、AttentionLayer、MoE。 |
| Attention 契约 | `attention/base.py`、`__init__.py`、`utils.py` | `BaseAttnBackend` 五个方法、metadata、HybridBackend、backend 注册与 graph 固定 buffer。 |
| Attention 实现 | `attention/fa.py`、`fi.py`、`trtllm.py` | 比较 KV 写入、page table 转换、workspace／plan、prefill／decode kernel、capture／replay。 |
| Engine | `engine/config.py`、`engine/engine.py`、`graph.py`、`sample.py` | 模型／KV／backend 装配，eager／graph，dummy padding，logits、采样、异步 D2H。 |
| 分布式 | `distributed/info.py`、`impl.py`、`__init__.py`；`scheduler/io.py` | TP rank、Gloo 控制面、torch／PyNCCL collective、all-reduce／all-gather。 |
| Python Kernel 桥 | `kernel/utils.py`、`index.py`、`store.py`、`radix.py`、`tensor.py`、`pynccl.py`、`moe_impl.py` | TVM FFI AOT／JIT、embedding 取行、KV 写入、Radix 比较、NCCL 与 MoE 包装。 |
| C++／CUDA | `kernel/csrc/jit/{index,store}.cu`；`kernel/csrc/src/{radix.cpp,tensor.cpp,pynccl.cu}` | warp copy、TensorMatcher、当前 CUDA stream、CPU key 比较与原生 NCCL。 |
| 头文件／Triton | `kernel/csrc/include/minisgl/{tensor.h,utils.h,utils.cuh,warp.cuh,nccl227.h}`；`kernel/triton/fused_moe.py` | shape／dtype／device 校验、launch／PDL、warp 原语、NCCL ABI、专家 GEMM 与 reduce。 |

## 术语表

| 术语 | 全书统一定义与工程口径 |
|-|-|
| TTFT / TPOT / ITL | TTFT 为请求发出到首段非空内容到达的延迟；TPOT 为首 token 后平均每输出 token 耗时；ITL 为相邻流式输出块间隔，更容易暴露中途停顿。 |
| Goodput | 单位时间内成功完成且同时满足全部预设延迟目标（如 TTFT 与 TPOT/E2E SLO）的请求速率（request/s）。 |
| Prefill / Decode | Prefill 并行处理未缓存输入并采出首个输出 token（常偏计算受限）；Decode 复用历史缓存逐步生成后续 token（小 batch 下常偏带宽受限）。 |
| Continuous Batching / Mixed Chunk | 连续批处理在轮次边界动态增删请求；Mixed Chunk 进一步允许同一次 `model.forward` 同时混装分块 Prefill 与在途 Decode token。 |
| Paged KV Cache / Radix Cache | 分页缓存通过页表将逻辑序列映射到离散物理页；Radix 树按压缩 token 前缀跨请求共享完整物理页，配合引用计数保护与叶节点 LRU 驱逐。 |
| Online Softmax / Split-KV | 利用局部最大值与对数指数和（`LSE`）结合律流式合并分块注意力；卡内 `split-KV` 借此让多个 CTA 并行扫描同一条长历史 KV。 |
| TP / PP / EP / DPA / CP | 五种并行维度：TP 切层内矩阵，PP 切连续层阶段，EP 按专家 ID 分片并通过 Dispatch/Combine 通信，DPA 在注意力层按请求分工消除 KV 副本，CP 沿同一长序列位置分摊上下文。 |
| CUDA Graph / Dual-Stream Overlap | CUDA Graph 捕获固定形状 Decode 计算图以省去逐算子 CPU 提交；双流 Overlap 将上一批 D2H 与后处理藏进当前批 GPU 执行窗口。 |
| W4A16 / W8A8 / FP8 KV | W4A16 仅压缩权重并在片上反量化；W8A8 同时量化权重与激活以调用低精度 Tensor Core；FP8 KV 独立压缩缓存体积，需配套校准 scale 与长上下文回归。 |
| Compressed FSM / XGrammar | 结构化约束解码技术：前者将唯一合法边合并为一次 Jump Forward；后者拆分 PDA 上下文无关/相关 token 缓存，并用紧凑 `int32` 位掩码在 GPU 原地屏蔽非法 logits。 |
| Speculative Decoding / EAGLE / MTP | 先低成本起草候选再由目标模型单次前向批量验证并无损恢复目标分布；EAGLE 复用高层特征轻量起草，MTP 使用预训练原生多 token 预测头，常配合 Tree Attention 验证整棵候选树。 |
| GQA / MLA / SWA / GDN | 四类状态压缩机制：GQA 多查询头共享少量 KV 头；MLA 用低秩潜向量加解耦 RoPE 与矩阵吸收压缩每 token 缓存；SWA 用环形缓冲只保留最近窗口；Gated DeltaNet 把整段历史折叠为固定大小的门控纠错状态矩阵。 |
| HiCache（L1 / L2 / L3） | 分层前缀缓存体系：`L1` 为 GPU HBM，`L2` 为实例 Host DRAM，`L3` 为 3FS/Mooncake 等外部存储；支持按页布局、排队预取与逐层事件重叠恢复。 |
| PD 分离 / Mooncake | 将 Prefill 与 Decode 部署到独立资源池并通过 RDMA 传输 KV，消除阶段互扰并允许非对称并行配比；Mooncake 以集群分布式 KV 池与 Conductor 联合路由实现跨节点复用。 |

## 当前 CLI 命令参考

依据 `server/args.py`、`scheduler/config.py`、`engine/config.py`、`engine/graph.py`；目标机先运行 `python -m minisgl --help`。

| 参数 | 默认值 | 作用 | 代码位置 |
|-|-|-|-|
| `--model` / `--model-path` | 必填 | HF repo ID 或本地目录 | `server/args.py` |
| `--tp-size` / `--tensor-parallel-size` | 1 | 单机张量并行卡数 | `DistributedInfo` |
| `--dtype` | auto | 权重与激活精度（auto 读取 HF config） | `parse_args` |
| `--max-running-requests` | 256 | 同时运行的请求上限（即 `page_table` 行数） | `EngineConfig.max_running_req` |
| `--max-prefill-length` / `--max-extend-length` | 8192 | 单轮 Prefill chunk 的 token 预算 | `SchedulerConfig.max_extend_tokens` |
| `--delay-prefill-wait-sec` | 0.0 | 新请求最长推迟等待秒数（`467f2eb` 扩展，0 关闭） | `Scheduler._admit_new_prefills` |
| `--delay-prefill-min-batch-size` | 0 | 触发放行的新请求排队数量阈值（0 关闭） | `Scheduler._admit_new_prefills` |
| `--delay-prefill-max-passes` | 0 | 新请求最多连续推迟的调度轮数上限（0 关闭） | `Scheduler._schedule_next_batch` |
| `--memory-ratio` | 0.9 | 权重与 KV Cache 合计可用显存比例 | `Engine._determine_num_pages` |
| `--num-pages` | 自动 | 显式覆盖物理 KV 页总数 | `num_page_override` |
| `--page-size` | 1 | 每页包含的物理 token 槽位数 | `Context.page_size` |
| `--attn` / `--attention-backend` | auto | `fa`、`fi`、`trtllm` 或 `prefill,decode` 组合 | `attention/__init__.py` |
| `--cache-type` | radix | 前缀缓存策略：`radix` 或 `naive` | `kvcache/__init__.py` |
| `--cuda-graph-max-bs` / `--graph` | 自动（160/256） | Decode 图捕获最大 batch，`0` 关闭 | `engine/graph.py` |
| `--num-tokenizer` / `--tokenizer-count` | 0 | 独立输入分词进程数（`0` 表示与 detokenizer 共用） | `server/launch.py` |
| `--moe-backend` | auto | MoE 算子后端（auto 默认选 `fused`） | `moe/__init__.py` |
| `--max-seq-len-override` | 无 | 覆盖模型配置的最大序列长度 | `EngineConfig.max_seq_len` |
| `--dummy-weight` | 关 | 使用随机权重调试形状与吞吐 | `Engine._load_weight_state_dict` |
| `--disable-pynccl` | 关 | 停用 TVM-FFI 包装器，回退至 `torch.distributed` NCCL | `Engine._init_communication` |
| `--host` / `--port` / `--shell-mode` | 127.0.0.1:1919 | 监听地址（`port+1` 用于分布式初始化）或交互式终端模式 | `ServerArgs` |
| `--model-source` | huggingface | 权重下载源，可选 `modelscope` | `parse_args` |

安装、在线服务与离线压测命令见本附录开头的〈[先把 Mini-SGLang 跑起来](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyTR3BJAUQLz8efORcVMoe4c)〉。编写启动脚本时请始终使用上表中的正式参数名（`--tp`、`--cache`、`--shell` 仅为当前 `argparse` 无歧义前缀缩写，后续新增同前缀参数时可能失效）。

<callout emoji="💡">
参数输入还会经过运行时调整：auto backend 依赖 GPU 架构，TRTLLM 会限制 page size，Graph 默认上限依赖模型加载前空闲显存。审查附录的参数表时，应结合 Attention 与 CUDA Graph 章节的调整规则。
</callout>

## 异构硬件对照：昇腾 NPU 分页 KV 写入算子接口

第 10 章讨论的 Mini-SGLang CUDA 后端通过原地更新 `_kv_buffer` 完成写入；迁移到昇腾 NPU 等异构加速器时，底层算子常采用函数式风格返回整层缓存池张量。xLLM 的 NPU GQA 后端展示了这一契约差异：`XPUStoreKVCachePageWrapper` 接收当前层 K/V 池、本轮 QKV 与块表索引，更新目标槽位后将整池引用一并返回。

```python
k_cache, v_cache, qkv_tensor = XPUStoreKVCachePageWrapper(
    k_cache=kv_cache[0],
    v_cache=kv_cache[1],
    qkv=qkv_tensor.unsqueeze(1),
    kv_len=attn_ctx.store_kv_lens,
    kv_block_table=attn_ctx.store_kv_ids,
)
```

这个实现按层独立分配 `kv_cache`，shape 为 `[2, num_blocks, Hkv, block_size, D]`。`2` 区分 K 和 V，`num_blocks` 是当前层、当前设备上整个池的物理块数，`block_size` 对应本章的 `P_page`；`Hq`、`Hkv` 是当前设备上的 Q/KV head 数，`D` 是 head dimension。层轴由各 attention 对象的 `self.kv_cache` 表达。与本节 Mini-SGLang 的统一存储相比，xLLM 的层组织方式和页内维度顺序有所不同。

| 返回值 | 逻辑 shape | 数据范围 |
|-|-|-|
| `k_cache` | `[num_blocks, Hkv, block_size, D]` | 当前层完整物理 K 池，包含各请求的历史 K、本轮写入及尚未使用的槽位。 |
| `v_cache` | `[num_blocks, Hkv, block_size, D]` | 当前层完整物理 V 池，覆盖范围与 K 池相同。 |
| `qkv_tensor` | `[T, 1, (Hq + 2Hkv) × D]` | 本轮进入 store 的 T 个 token，末维依次排列全部 Q head、全部 K head、全部 V head。后续 attention 继续消费这份 QKV。 |

cache 按 `[物理块号, KV head, 块内位置, head 通道]` 索引。对本轮第 t 个 token，`store_kv_lens[t]` 给出请求内的逻辑位置；`store_kv_ids[t]` 给出所属请求的块表。以下伪代码描述有效 token 的写入地址：

```python
position = store_kv_lens[t]
logical_block = position // block_size
offset = position % block_size
physical_block = store_kv_ids[t, logical_block]

k_cache[physical_block, :, offset, :] = K_new[t]
v_cache[physical_block, :, offset, :] = V_new[t]
```

`K_new[t]` 和 `V_new[t]` 分别含本轮这个 token 的 `[Hkv, D]` 数据。例如，`block_size = 16`，某请求的块表为 `[7, 2]`，本轮两个新 token 的逻辑位置为 15 和 16。写入只触及物理块 7 的槽位 15，以及物理块 2 的槽位 0；返回的 K/V cache 仍能索引池中的全部 `num_blocks` 个块。已有历史 KV 保留在原位置。尚未使用的槽位可能保留初始化值或旧数据，有效读取范围由块表和序列长度共同约束。

**返回整个池，不会让某条请求的 KV 自动变成按 token 顺序排列的连续张量。** 请求可以使用物理块 7、2、12，attention 仍通过块表读取连续的逻辑序列。创建时的一次 `torch.zeros` 加 `view` 得到连续的大张量；单条请求分配到的块号可以不相邻。这两种连续性分别描述缓存池布局和请求地址映射。

返回值的覆盖范围与底层存储是否复用是两个问题。Python wrapper 只转交算子并接收三个输出，无法据此确认输出与输入的 storage 是否相同，也不能推断运行时复制了整个池。别名关系和实际拷贝行为需要检查原生内核或在 NPU 上验证；meta 定义中的 `empty_like` 仅用于输出元数据推导。

源码依据：[xpu_attn.py：缓存分配](https://code.byted.org/data/xLLM/blob/c3e4a837270c63fdfe5e2ee20db09c890c0e8757/xllm/backend/inferencer/torch/layers/x2/attention/backends/gqa/xpu_attn.py#L132)、[分页写入与后续读取](https://code.byted.org/data/xLLM/blob/c3e4a837270c63fdfe5e2ee20db09c890c0e8757/xllm/backend/inferencer/torch/layers/x2/attention/backends/gqa/xpu_attn.py#L674)、[每层持有返回的缓存池](https://code.byted.org/data/xLLM/blob/c3e4a837270c63fdfe5e2ee20db09c890c0e8757/xllm/backend/inferencer/torch/layers/x2/attention/gqa_attn.py#L181)。形状及算子转发另核对了锁定依赖 `bytedxpu_ops 1.0.0.49` 的 Ascend wrapper 和 meta 定义。

## 版本维护

基线是 `main@9a91cfafe754aa85daee49998176275667eb58f2`。升级先比较契约：入口看 `args.py／launch.py／README`；运行时看 `Req／Batch／Context` 与 `_prepare_batch`；资源看页表、KV、prefix cache、准入和释放；性能重新记录软硬件、模型 revision 与 workload。

```bash
git rev-parse HEAD
git diff --stat 9a91cfafe754aa85daee49998176275667eb58f2..HEAD -- python/minisgl README.md docs
git diff 9a91cfafe754aa85daee49998176275667eb58f2..HEAD -- python/minisgl/core.py python/minisgl/server/args.py
python -m minisgl --help
```

graphify 只找相邻文件和候选调用者，不能替代源码。图有悬空端点、自环或折叠边时，不得把边当事实。更新后继续标记“源码已核／已运行／待验证”。

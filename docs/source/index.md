# LLM 推理引擎

<callout emoji="💡">
**阅读路线（全书 6 篇 24 章 + 2 篇附录）：**本文紧随一条在线请求，从单步生成语义走到 GPU 执行与显存记账，穿透 Mini-SGLang 核心源码后，再向生产版 SGLang 与超大规模集群系统逐层推开。
- **第一篇 · 推理基础与性能标尺（第 1～4 章）：**依次建立自回归生成与单次前向闭环（含 RoPE 相对位置旋转）、GPU 执行与 Roofline / Online Softmax 瓶颈判断、全书唯一的 KV Cache 容量账，以及 Benchmark 延迟与 Goodput 口径。
- **第二篇 · Mini-SGLang 核心运行时走读（第 5～12 章）：**先搭起静态进程骨架并跑通示例请求 `R0`，再沿 Scheduler（含延迟准入与分块 Prefill）、KV 与 Radix 树、单机 TP、模型执行与采样、Attention 三后端（含卡内 `split-KV`）、CUDA Graph 与双流重叠，一路追到自定义 Kernel 与全系统消融实验。
- **第三篇 · 走向生产引擎：前后端协同、调度与并行（第 13～15 章）：**对照 SGLang 语言模型程序、模块映射与网关亲和路由，深入单机 Mixed Chunk 与 NanoFlow 设备内异构流水线，并统一对比 PP、EP、DP Attention 与 Context Parallel。
- **第四篇 · 低精度计算与解码层加速（第 16～19 章）：**覆盖量化格式与计算路径、推理确定性与逐层精度对齐验收、结构化约束解码（Outlines / Jump Forward / XGrammar 位掩码）以及投机解码（独立草稿 / EAGLE / MTP 与 Tree Attention 验证）。
- **第五篇 · 新型状态、多级缓存与集群架构（第 20～24 章）：**先推导 MLA 低秩矩阵吸收、SWA 环形窗口与 Gated DeltaNet 固定状态递推，再进入单机三级恢复（HiCache）、跨机 PD 分离与 Mooncake 全局池、DeepSeek V4.1 Flash 综合案例，最后以长上下文调优、AIPerf 容量压测与服务端 Profiling 收口。
- **第六篇 · 附录与参考（第 25～27 章）：**附录 A 一站式提供 Mini-SGLang 安装、CLI 参数、OpenAI 兼容边界、排障表与源码索引；附录 B 给出生产版 SGLang 插件扩展骨架；文末集中收录分类文献与证据边界。
系统学习建议顺读；直接读源码可从第 5 章切入并配合附录 A 的[分阶段学习路线](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyg6IsklSdDD1r74mwD53Rgc)；线上排障与容量验收可直接跳至第 24 章与附录 A。
</callout>

本文面向阅读与调优推理引擎的工程师。Mini-SGLang 源码走读以官方提交 `9a91cfa` 为统一主基线（本地扩展提交 `467f2eb` 的延迟准入单列说明），生产版 SGLang 模块对比、量化、DPA、HiCache 与插件机制各自锁定对应源码提交。原理推导、静态源码事实与论文实测结果严格区分各自的适用条件。

```{toctree}
:maxdepth: 2
:caption: 目录

chapters/01-chapter
chapters/02-gpu-roofline
chapters/03-kv-cache-k-v
chapters/04-goodput
chapters/05-mini-sglang
chapters/06-scheduler-batch
chapters/07-kv-cache
chapters/08-tensor-parallel-rank
chapters/09-batch-token
chapters/10-attention-backend-q-k-v-paged-kv-cache
chapters/11-cuda-graph-decode
chapters/12-kernel
chapters/13-mini-sglang-sglang
chapters/14-mixed-chunk-nanoflow
chapters/15-pp-ep-dp-attention-context-parallel
chapters/16-chapter
chapters/17-chapter
chapters/18-fsm-jump-forward-xgrammar
chapters/19-eagle-mtp
chapters/20-mla-swa-gated-deltanet
chapters/21-kv
chapters/22-pd-kv-mooncake
chapters/23-deepseek-v4-1-flash
chapters/24-profiling
chapters/25-a-mini-sglang
chapters/26-b-sglang
chapters/27-chapter
```

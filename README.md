# LLM 推理引擎

中文 | [English](README.en.md)

## [在线阅读本书](https://fangpin.github.io/llm-infer-book/)

沿着一次在线请求，理解大模型推理引擎如何工作。本书从自回归生成、GPU 执行和 KV Cache 入手，结合 Mini-SGLang 源码走读运行时，再讨论 SGLang 的生产调度、分布式并行与集群架构。

## 阅读这本书

适合希望阅读、开发或调优推理引擎的工程师。全书包含 24 章正文、2 篇附录和参考资料，正文目前为中文。

- **从基础开始**：从[第 1 章](https://fangpin.github.io/llm-infer-book/chapters/01-chapter.html)顺序阅读，先理解单步生成、GPU 瓶颈和 KV Cache，再进入运行时实现。
- **对照源码学习**：从[第 5 章](https://fangpin.github.io/llm-infer-book/chapters/05-mini-sglang.html)进入 Mini-SGLang，沿第 5 至 12 章追踪请求、调度、模型执行和 Kernel，配合[附录 A](https://fangpin.github.io/llm-infer-book/chapters/25-a-mini-sglang.html)查阅安装步骤、参数和源码索引。
- **排查性能问题**：先用[第 4 章](https://fangpin.github.io/llm-infer-book/chapters/04-goodput.html)统一延迟、吞吐与 Goodput 的度量口径，再阅读[第 24 章](https://fangpin.github.io/llm-infer-book/chapters/24-profiling.html)的容量压测和 Profiling 方法。

| 章节 | 内容 |
| --- | --- |
| 第 1 至 4 章 | 自回归生成、GPU 执行模型、Roofline、KV Cache 与性能指标 |
| 第 5 至 12 章 | Mini-SGLang 源码走读：Scheduler、Radix Cache、Tensor Parallel、Attention Backend、CUDA Graph 与自定义 Kernel |
| 第 13 至 15 章 | SGLang 前端与运行时、Mixed Chunk、NanoFlow，以及分布式并行 |
| 第 16 至 19 章 | 量化、推理确定性与精度对齐、结构化约束解码、投机解码 |
| 第 20 至 24 章 | MLA、SWA、Gated DeltaNet、多级缓存、PD 分离、Mooncake、DeepSeek 案例与生产调优 |
| 附录与参考 | Mini-SGLang 实战手册、SGLang 插件接入和参考资料 |

源码分析以各章注明的提交为准。原理推导、示例计算与论文实测结果分别说明适用条件，便于对照自己的模型、硬件和请求负载阅读。

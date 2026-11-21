# LLM Inference Engines

[中文](README.md) | English

## [Read the Book Online](https://fangpin.github.io/llm-infer-book/)

Follow an online request through an LLM inference engine. The book starts with autoregressive generation, GPU execution, and the KV cache, then uses Mini-SGLang source walkthroughs to explain the runtime before moving on to production scheduling, distributed parallelism, and cluster architecture in SGLang.

## Reading the Book

This book is for engineers who want to understand, develop, or tune inference engines. It contains 24 chapters, two appendices, and references. The book itself is currently written in Chinese.

- **Start with the fundamentals**: Read from [Chapter 1](https://fangpin.github.io/llm-infer-book/chapters/01-chapter.html) to build an understanding of single-step generation, GPU bottlenecks, and the KV cache before exploring runtime implementations.
- **Read alongside the source**: Start with Mini-SGLang in [Chapter 5](https://fangpin.github.io/llm-infer-book/chapters/05-mini-sglang.html). Chapters 5-12 follow requests through scheduling, model execution, and kernels. Use [Appendix A](https://fangpin.github.io/llm-infer-book/chapters/25-a-mini-sglang.html) for installation instructions, configuration options, and the source index.
- **Investigate performance**: Begin with [Chapter 4](https://fangpin.github.io/llm-infer-book/chapters/04-goodput.html) to establish consistent definitions of latency, throughput, and goodput, then turn to [Chapter 24](https://fangpin.github.io/llm-infer-book/chapters/24-profiling.html) for capacity testing and profiling.

| Chapters | Topics |
| --- | --- |
| 1-4 | Autoregressive generation, GPU execution, the Roofline model, KV cache, and performance metrics |
| 5-12 | Mini-SGLang source walkthroughs: scheduler, radix cache, tensor parallelism, attention backends, CUDA Graphs, and custom kernels |
| 13-15 | SGLang frontend and runtime, Mixed Chunk, NanoFlow, and distributed parallelism |
| 16-19 | Quantization, inference determinism and numerical validation, constrained decoding, and speculative decoding |
| 20-24 | MLA, SWA, Gated DeltaNet, tiered caching, prefill/decode disaggregation, Mooncake, DeepSeek case studies, and production tuning |
| Appendices and references | Mini-SGLang practical guide, SGLang plugin integration, and reference material |

Source analysis is tied to the commits identified in each chapter. Theoretical derivations, worked examples, and measurements reported in papers are distinguished, with their assumptions stated so readers can assess their relevance to a particular model, hardware setup, and workload.

# 推理确定性与精度对齐验收

```{contents} 本页目录
---
depth: 2
local: true
---
```

低精度量化、算子融合、动态合批与多卡并行都会改变浮点运算路径。工程实践中常遇到两类紧密相连的问题：线上排障时，“为什么相同的模型、参数和输入在两次请求中给出了不同回答？”；迁移引擎、切换后端或上线量化模型时，“多大的数值偏差属于正常浮点舍入，多大的偏差说明实现有 Bug 或质量不可接受？”。

本章将这两条线索合为一章：前半部分拆解在线推理非确定性的来源与逐 token 复现条件；后半部分给出推理引擎迁移、低精度量化与 LoRA 合并的逐层数值对齐方法及验收标准（注：这里的“对齐”指计算数值与模型效果对齐，不同于 RLHF/DPO 的人类偏好对齐）。

## 同样的请求为何输出不同

同样的模型、生成参数和用户输入仍可能产出不同文本，根源主要落在两处：采样阶段本身引入了随机性，或实际前向计算与上下文条件并未完全固定。顺着 **token ID → hidden state → logits → token ID** 的执行链，可以逐段收窄排查范围。

<callout emoji="💡">
**temperature=0 不能单独保证复现。**它通常让解码器进入 greedy 分支，消除随机抽样；要让输出逐 token 一致，还需要相同的完整输入、模型部署和确定性的计算过程。具体参数支持与 greedy 判定，以模型服务和推理引擎的实际实现为准。
</callout>

### 相同采样参数仍会抽到不同 token

Temperature、Top-k 和 Top-p 规定如何处理候选分布，并不规定一次抽样的结果。即使每一步的 logits 完全相同，只要使用随机采样，两次生成也可能选择不同 token。前文的概率示例中，Top-k=2 后两个候选约为 0.571 和 0.429；重复从这组分布抽样，不必每次都选第一个。

Greedy 直接选最大 logit。在 logits 和并列分数处理规则一致时，选择结果应一致；如果实现把低温采样近似当作 greedy，仍应检查它是否真的执行 argmax。

### 细小数值差异可能改变最大 logit

浮点运算需要舍入，通常不严格满足结合律。对某些浮点数，先算 `(a+b)+c` 与先算 `a+(b+c)` 会得到不同结果。并行归约顺序、算子实现，以及是否使用 FMA，都可能改变中间计算结果。FMA 对乘加只舍入一次，分开的乘法与加法通常舍入两次。

GPU 调度或分布式通信顺序是否影响结果，取决于算法实现。不同硬件和计算精度也可能产生数值差异。**只有这些差异改变了候选 token 的排名，才会改变 greedy 的选择；有数值差异不等于一定有文本差异。**

排名前两位的 logits 非常接近时，微小扰动就可能使 argmax 改变。由此可见，greedy 去掉了抽样随机性，却没有自动消除前向计算的非确定性。

### 模型名称和用户输入不足以固定请求

| 差异来源 | 实际变化 | 应核对的对象 |
|-|-|-|
| 运行环境与动态 batching | 两次请求可能进入不同 GPU、batch 形状或算子路径。batch 变化是否影响输出，取决于实现是否保证 batch invariance。 | 硬件、引擎与依赖版本、batch 组成、并行策略 |
| 模型部署与网关路由 | 相同模型别名可能经历版本更新、灰度路由或自动 fallback。仅凭输出差异，不能断定实际换了模型。 | 实际处理请求的模型版本、部署信息、路由与 fallback 记录 |
| 计算精度与量化配置 | 同名模型可能使用不同权重、激活或 KV Cache 精度，产生不同数值结果。 | 权重与 tokenizer 版本、量化配置、启动参数、运行时精度 |
| 完整上下文 | 用户文字相同，但 system/developer prompt、历史对话、记忆、工具定义或自动摘要不同。 | 最终发送给模型的完整请求，以及序列化后的 token 输入 |
| 检索与工具结果 | RAG 文档、排序或外部数据更新；Agent 工具结果的到达与追加顺序也可能不同。 | 检索快照、工具返回值、上下文组装顺序 |
| 响应后的处理 | 应用过滤、格式化、JSON 修复、重试或多候选选择改变最终展示；流式连接中断可能只展示部分响应。 | 原始响应、停止原因、重试记录与最终展示结果 |

某些带容量上限丢弃（token dropping）的 MoE 实现可能受同批竞争影响，但这取决于具体路由策略，**MoE 架构本身不等于输出不稳定**。量化配置的核对口径见前一章〈[量化与低精度推理](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyjNEU2htOhQ616pcxaA5Ceh)〉的“配置比模型名更可靠”一节。

### 固定 seed 仍需固定随机数使用方式

Seed 用于初始化伪随机数生成器。复现抽样结果还要求相同的随机状态、随机数消费顺序与采样实现。服务启动时只设置一次全局 seed，不会让每个后续请求自动从同一随机状态开始；此前请求已经消费的随机数也会影响后续抽样。

第 9 章〈[模型执行与采样：从 Batch 到 token](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyTjwgnLCFNIX7iKHS0V2TVh)〉的 Sampler 小节展示了一个典型源码边界：Mini-SGLang 使用 `torch.manual_seed(42)` 固定进程级全局种子，但没有 per-request seed；混合批中的 greedy 请求还会以 `1e-6` 低温进入随机采样 kernel，最大 logits 并列时可能选出不同 token。因此排查复现问题必须核对真实代码分支，不能只看请求参数。

平台支持 per-request seed 时，它通常能提高复现性，但不应据此承诺跨硬件、跨版本或不同执行路径的严格一致。vLLM 的复现文档也把硬件和引擎版本保持一致列为条件，并区分离线确定性调度与在线 batch invariance。

### 一个 token 的差异会影响后续生成

自回归生成会把已经生成的 token 带入下一轮。某一步选择不同，后续条件概率分布就会变化，差异可能逐步扩大，因此很小的初始扰动也可能变成整段不同的答案。推理型模型生成的中间推理序列同样可能不同，即使这部分内容没有展示，仍可能影响最终答案。

### 复现检查先找第一个不同的 token

| 检查对象 | 固定或记录什么 | 能排除什么 |
|-|-|-|
| 模型与输入 | 固定版本或快照；保存完整请求、所有生成参数、检索与工具快照。自部署时核对权重和 tokenizer。 | 别名更新、隐藏上下文、数据变化 |
| 解码与随机状态 | 在服务支持时使用 greedy 或 per-request seed；检查实际 argmax 分支、并列处理和全局随机状态。 | 抽样与参数解释差异 |
| 计算与调度 | 自部署时固定硬件、精度、引擎和依赖版本，采用确定性算子；核对 batch invariance。仅开启确定性算子不足以保证整个系统复现。 | 数值路径与 batch 变化 |
| 原始响应与展示 | 保存原始响应、停止原因、实际模型与重试记录。先比较生成的 token 序列，再比较应用最终展示。 | 截断、修复与后处理差异 |

如果输入和配置已经对齐，优先定位第一个不同的 token；能够获取 logits 时，再比较该步的候选排名与采样随机状态。这样可以区分“同一分布抽到了不同结果”和“计算出的分布已经不同”。无法获取服务端 logits、batch 或部署信息时，应明确剩余不可观测条件，不能把差异统一归因于采样。

**逐 token 完全一致与结论一致是不同的目标。**固定权重、完整 token 输入、全部执行状态和解码规则，并使用确定性的实现时，输出应一致。在线服务通常没有固定全部条件；业务验收可按需要分别检查文本一致、结构合法和结论一致。

**依据与延伸阅读：**[NVIDIA：Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)说明浮点结合律、计算顺序与 FMA 的数值影响；[vLLM：Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/)说明随机种子、调度、batch invariance 和硬件／版本边界。资料于 2026 年 11 月 6 日核对；具体服务是否支持 temperature、seed 或确定性配置，应以对应版本的接口文档为准。

## 对齐目标决定验收标准

| 对齐目标 | 核心检查内容 | 典型推理场景 |
|-|-|-|
| 数值一致 | 逐层中间张量与有效位置 logits 在容差内贴近 reference | 新模型接入、算子替换、推理引擎跨框架迁移 |
| 行为接近 | 下一 token 概率分布、KL 散度、Top-1 一致率与候选排序稳定 | 混合精度路径、Attention 后端切换、CUDA Graph 与动态合批验证 |
| 效果可接受 | 困惑度与困难推理基准满足预先固定的退化预算 | W4A16、W8A8、FP8/FP4 权重与 KV Cache 量化验收 |

跨硬件、跨融合算子的浮点计算通常不要求逐 bit 相等；低精度量化主动引入近似误差，应区分“量化算法本身的理论损失”与“部署算子写错契约的工程误差”。

## 比较前先锁死计算契约

Reference 应优先选择已验证的朴素高精度实现（如 Hugging Face Transformers eager 路径）。若同时更换引擎、dtype 与 Attention 后端，必须先在相同未量化 dtype（如 BF16 或 FP32）下对齐框架，再单独打开算子优化和低精度量化。

| 核对维度 | 必须锁定的条件 |
|-|-|
| 模型权重 | 同一 checkpoint 与 revision；核对 QKV/gate-up 合并顺序、转置方向及 `tie_word_embeddings` 共享关系 |
| 结构配置 | RoPE 基频与缩放参数、RMSNorm epsilon、激活函数、GQA/MQA 分组比及滑窗长度一致 |
| 输入编码 | 直接保存并喂入同一组 `input_ids`，排除 tokenizer 版本、chat template 与特殊 token 解析差异 |
| 位置与掩码 | `positions` 偏移、causal mask、padding 屏蔽及有效输出下标一致 |
| 精度配置 | 分别记录权重、激活、KV Cache、Softmax/Norm 归约累加器的 dtype（各格式指数与尾数位宽见前一章） |
| 运行环境 | eval 模式、GPU 架构、驱动、依赖库版本、batch 形状、TP/CP 并行度与 TF32 开关 |

## 用相同 token 序列定位前向误差

### Teacher forcing 保持后续输入可比

先给 reference 与待测实现相同的完整 token 序列，让双方在每个位置预测下一个 token，再比较对应位置的 logits。这种 teacher forcing 比较不会因生成结果分歧而改变后续输入。Causal LM 的 logits 通常预测下一个 token，必须核对位置和 labels 的对应关系，排除 padding、忽略的 label 和无效位置。

自由生成适合最终行为验收，却不适合作为第一轮数值定位手段。某一步选出的 token 不同，后续上下文就不同，此后输出差异无法单独归因于数值误差。即使使用 greedy，排名接近的候选也可能被小误差交换；随机采样还会引入额外波动。

### 数值误差与预测分布需要一起看

| 指标 | 用途 | 解释边界 |
|-|-|-|
| 最大绝对误差 | 发现局部异常与极端偏差 | 容易受单个极值影响，不能单独概括整体 |
| 平均绝对误差、误差分位数 | 观察整体误差与尾部误差 | 平均值可能掩盖少数严重异常；分位数需注明算法 |
| 相对 L2 误差 | 按 reference 整体尺度衡量偏差 | reference 范数接近零时需谨慎解释 |
| 余弦相似度 | 观察整体方向是否接近 | 方向接近仍可能存在尺度偏差，不能单独判定通过 |
| token 概率分布的 KL 散度 | 观察预测分布变化 | 先固定词表、位置和概率转换口径，并注明 KL 方向 |
| Top-1 一致率与候选分差 | 检查是否选出相同 token，解释排名翻转 | 候选分差很小时，小误差也可能改变 Top-1 |

逐元素比较常用以下条件：

$$|x_{\mathrm{test}}-x_{\mathrm{ref}}|\leq\mathrm{atol}+\mathrm{rtol}\,|x_{\mathrm{ref}}|$$

整体相对 L2 误差可写为：

$$E_{\mathrm{rel}}=\frac{\|x_{\mathrm{test}}-x_{\mathrm{ref}}\|_2}{\|x_{\mathrm{ref}}\|_2+\epsilon}$$

atol 控制接近零时的绝对误差，rtol 随 reference 尺度变化，epsilon 用于避免分母为零。接近零的元素不宜只用相对误差判断。容差取决于 dtype、张量尺度、模型深度、算子和正常实现间的波动，没有适用于所有 LLM 的统一数值。

若 reference 的前两名 logits 为 10.001 和 10.000，微小误差就可能交换排名。生成文本不同不能直接证明实现错误，文本相同也不能证明中间数值已经对齐。随机采样的行为验收，应在相同采样配置下比较分布或重复评测结果。

### 逐层比较找到误差开始扩大的位置

最终 logits 超出容差时，沿前向链保存中间张量。先比较 Embedding 和若干层输出；层数多时，用二分法缩小范围，再检查该层内的 Norm、Q／K／V、RoPE、Attention、残差和 MLP，最后检查 Final Norm 与 LM Head。首处差异是排查入口，是否构成缺陷仍需结合算子正常误差判断。

| 异常位置 | 优先检查 |
|-|-|
| Embedding 已不同 | token ID、词表映射、权重加载与参数共享 |
| RoPE 后偏差明显 | position、旋转维度排列、RoPE scaling 配置 |
| Attention 后偏差扩大 | mask、缩放系数、GQA／MQA head 映射、Softmax 和归约精度 |
| Norm 后偏差扩大 | epsilon、归约与累加精度 |
| MLP 后偏差扩大 | 激活函数、gate／up 权重映射、量化参数 |
| 只有最终 logits 异常 | Final Norm、LM Head、词表分片与映射 |

诊断时可局部提高 Norm、Softmax 或归约计算的精度，观察误差是否收敛。定位完成后，还需在实际准备使用的优化配置下重新验收，不能只验证临时高精度路径。

## 推理迁移分别验证 Prefill、Decode 和 Batch

推理框架迁移先对齐相同 dtype 的基础前向，再验证 KV Cache、batch 和优化 kernel。这样能区分模型实现误差、缓存错误与执行优化带来的数值变化。

| 执行路径 | 最小比较实验 | 重点检查 |
|-|-|-|
| 完整前向／Prefill | 相同 token、position 和 mask，比较有效位置 logits | 权重映射、模型配置、Attention 与 MLP |
| 带 KV Cache 的 Decode | 对相同前缀，比较每次重算完整前缀与复用 cache 的下一步 logits | cache 地址、有效长度、dtype 与位置偏移 |
| Batch／padding | 同一请求单独执行与放入不同长度 batch 执行 | padding、mask、position、请求隔离和索引 |
| 优化 kernel | 在相同输入下，对比基础路径和 fused／Flash Attention 等路径 | 累加精度、归约顺序和优化配置 |
| 端到端生成 | 固定采样配置，覆盖 EOS、长度上限和代表性任务 | 生成行为、质量和请求终态 |

如果 Prefill 正常，带 cache 后异常，优先核对 K／V 布局、head 对应关系、写入位置、有效长度、decode mask、position 偏移和请求间隔离。KV 量化还要核对 scale 与反量化。Cache 基础契约见[“KV Cache 用显存换取历史 K/V 复用”](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyeCddmQDtIOIJdz5JSRA1pg)。

Batch shape 变化可能让后端选择不同 kernel，因此单独运行与 batch 运行存在微小差异，并不自动说明实现有错。明显偏差则应先检查掩码、位置和 cache 索引。Mini-SGLang 的具体消融与证据要求见[第 12 章全系统消融实验](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyUJVJws0QtQpXf6hF3NdEmd)；其中逐 token 一致可作为额外行为目标，但遇到接近的候选时，仍需结合 logits 和候选分差诊断。

## 量化分别评估方案损失与实现误差

W4A16、W8A8、FP8 和 KV Cache 量化都可能改变数值。前章已说明格式、scale 和性能路径，本节把验收拆为两组比较，避免将正常量化误差和部署错误混为一谈。

| 比较关系 | 要回答的问题 | 验收依据 |
|-|-|-|
| 高精度模型与量化参考实现 | 量化方案本身损失了多少精度 | 逐层误差、预测分布、困惑度与任务退化预算 |
| 量化参考实现与实际部署 kernel | 部署是否正确实现了同一量化契约 | 格式、打包与 scale 一致，输出误差满足该实现的容差 |

量化契约应核对对称／非对称方式、per-tensor／per-channel／per-group、group size 与分组维度、scale 和 zero point、舍入与截断、权重打包顺序，以及激活是否动态量化。FP8 还要确认具体格式与 scale 管理。同叫 INT4，不代表权重和 kernel 可以互换；W4A16 也不描述 KV Cache 的精度。

需要校准的方案，应让数据覆盖实际语言、任务与长度分布。短文本校准通过，不代表长上下文通过。量化后的 KV 会被后续生成反复读取，长上下文和长输出要单独评测。原有公开结果与困难任务回归要求见[前章“质量回归要覆盖困难任务”](https://bytedance.larkoffice.com/wiki/EZcawGzFrirc8bkkvuqclJCpnXd#doxmyqSIZNA8xoJMnnfajgUH5ig)。

定位个别敏感层后，可尝试更细分组、处理 outlier，或让敏感层保留高精度，同时衡量元数据、kernel、显存和性能成本。QAT 会调整权重以适应量化，应按最终模型效果验收，通常不要求权重或中间张量与原模型相同。

## 多卡并行与 LoRA 合并的数值核对重点

单卡对齐通过后，切入分布式推理或加载适配器时还需单独检查归约与合并边界：

- **硬件累加路径与 TF32：**即使输入标为 FP32，部分框架默认允许 Tensor Core 走 TF32 乘加或低精度归约；诊断微小偏差时可临时关闭 TF32 与融合算子定位来源，最终验收仍须回到真实上线配置。
- **张量并行（TP）与上下文并行（CP）：**重点核对 Q/K/V 与 KV head 复制映射、Row Parallel 的 `all_reduce` 求和位置、`bias` 是否在多卡求和前被重复相加、词表分片边界，以及 CP/split-KV 局部 Softmax 合并时的 `LSE` 重缩放精度。正常分片会改变浮点求和顺序并带来微小舍入波动，而分片错位或重复加 bias 则会造成显著发散。
- **LoRA 在线旁路与离线合并：**在线计算 $y=Wx+sB(Ax)$ 与离线合并权重 $W^\prime=W+sBA$ 数学等价，但矩阵乘法结合顺序与权重落盘 dtype（如合入 BF16 或重做 INT4/FP8 量化）会引入附加误差；验收时需同时比对合并前后的 teacher-forcing logits 与目标任务指标。

## 验收报告保留容差与首处分歧证据

容差阈值应在正式测试前根据参考实现间的正常波动与业务退化预算预先固定，不能在看到失败后随意放宽：

| 记录项 | 验收报告应留存的证据 |
|-|-|
| 实验基线与环境 | reference 实现、checkpoint revision、固定 `input_ids`、GPU 型号、kernel 版本、各类 dtype 与并行开关 |
| 前向数值定位 | Prefill/Decode/Batch 各路径的 `atol/rtol`、相对 L2、KL 散度、超阈值层位置及排名前二候选分差 |
| 任务质量与性能 | 固定评测集、prompt 模板与采样配置下的任务得分波动、失败样例，以及同条件下的 TTFT、ITL、吞吐与显存 |

<callout emoji="✅">
**本章核心自查：**能解释为何定位前向误差第一轮必须使用固定 `input_ids` 的 teacher forcing 而非自由生成，并说清如何用两组对照把“量化算法本身的理论损失”与“部署 kernel 的实现 Bug”拆开。
</callout>

数值确定性与精度验收方法明确后，接下来两章进入解码层加速：先看结构化约束解码如何筛除非法候选并跳过确定字符串，再看投机解码如何用一次前向验证推进多个候选 token。

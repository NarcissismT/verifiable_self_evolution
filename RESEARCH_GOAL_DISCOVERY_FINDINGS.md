# 自主研究目标发现：文献核查汇总

本文档汇总三轮定向文献检索的结果，主题是 deep research agent 在「面对自然、模糊提问时不知道该补什么背景、比较什么对象、哪些结论需要因果分析、何时该反问」上的能力缺口，以及修复该缺口的训练路线可行性。

起点是 arXiv:2608.02163（From Simple QA to Deep Research，Can Wang 等）。该论文的 q_h / q_o / q_a 三种 query 形式共享同一套 rubric 与 DAG，为「特权信息教师 → 无提示学生」提供了天然的实验载体，这是本轮调研的主要动机。

检索日期 2026-08-19。三轮分别为 110 / 109 / 106 个 agent，合计抽取 392 条 claim，其中 75 条进入 3 票对抗验证，40 条存活。

## 证据强度约定

本文档中每条结论都标注来源与强度，含义固定如下：

| 标注 | 含义 |
| --- | --- |
| `3-0` / `2-1` | 通过 3 票对抗验证，票型如标注 |
| `检索未验证` | 一手检索片段，未进入验证队列（预算限制），引用前需自行核对原文 |
| `被否` | 进入验证但被推翻，不作为证据使用 |
| `推断` | 从其他已验证结论推导，非任何单篇文献的结论 |

未标注即为 `3-0`。凡涉及本项目决策的关键结论，标注一律显式给出。

## 一、缺陷的性质

### 1.1 瓶颈是 rubric formation，不是推理力

RubricBench（arXiv:2603.01562v1，CityU + Tencent Hunyuan + Mila + MBZUAI）做了固定 backbone、prompt、解码参数，仅替换 rubric 来源的受控实验。self-generated 与 human rubric 的绝对准确率差在 7 个模型上为 +22.1 至 +28.3 个百分点。

| 模型 | 自生成 rubric | 人工 rubric | 差值 |
| --- | --- | --- | --- |
| Gemini-3-Pro | 60.4 | 82.5 | +22.1 |
| Qwen3.5-Plus | 59.3 | 84.2 | +24.9 |
| DeepSeek-v3.2 | 57.8 | 84.9 | +27.1 |
| GPT-5.1 | 54.6 | 82.9 | +28.3 |
| GPT-OSS-120B | 56.4 | 84.7 | +28.3 |
| Gemini-3-Flash | 58.0 | 85.3 | +27.3 |
| GPT-4o-mini | 46.7 | 73.4 | +26.7 |

原文断言 the primary limitation in current evaluation is not reasoning capacity, but rubric formation，并给出 they systematically fail to autonomously induce the necessary evaluation criteria。

归因不止于作者断言。Appendix D.1 的人类对照实验中，人类评估者用人工 rubric 得 92.0%，换成模型生成的 rubric 降至 61.0%，即劣质 rubric 同等程度地拖垮人类，排除了「评估者推理力不足」这一竞争解释。机制侧：自动 rubric 方法 hallucination rate 超过 70%，平均漏掉 13 条以上关键约束，低必要性条目占比 17.9% 对人类的 10.1%。

该 gap 从 GPT-4o-mini 到 GPT-5.1 / Gemini-3-Pro 全模型族不收窄（最强的 Gemini-3-Pro 反而最小，仅有轻微随能力收窄迹象）。等待更强基座不解决问题。

需要打折的地方：+26pp 是 1,147 条刻意对抗构造的 hard pair 上的量，vanilla 基线普遍贴近随机，非对抗分布上大概率显著更小，不能当常数引用。即便给人工 rubric 也只到约 85%，因此 rubric formation 是主导瓶颈但非唯一瓶颈。「与 q_h / q_o 同构」是本文档的类比推断，RubricBench 全文未提及 arXiv:2608.02163。

### 1.2 掉分的构成需要限定

论文的 rubric 满分按 evidential / analytical 拆分，均值分别为 0.52 与 0.48。以 GPT-5.6 Terra 为例，q_h 是 0.86（0.41 + 0.45），q_o 是 0.67（0.30 + 0.37）：E 绝对掉 0.11，A 绝对掉 0.08；换成相对损失，E 掉 21%，A 掉 17%。

论文原文说的是提示变弱时强弱模型的差距更多由 analytical 驱动，指的是模型间差距的构成，不是单模型绝对掉分的构成。前沿模型自身的绝对掉分 E 侧更多。MiniMax-M2.5 的形状更极端：q_o 为 0.42（0.18 + 0.24），E 从 0.30 掉到 0.18，降幅 40%，比 A 更狠。

工程含义：不要把病灶当成纯 analytical。q_o 下连「该检索什么」都在退化，因为不知道该比较哪些对象会直接导致 evidential checkpoint 也拿不到。奖励设计上不应手动给 analytical 加权；DAG 的结构化权重传播（叶节点等权、反向拓扑序向父节点传递、归一化）本身就让支撑更多下游结论的前序步骤拿到更多权重，而 evidential 节点通常正是那些前序步骤。默认权重是对的。

### 1.3 交互澄清是当前性价比最高的解法

IDRBench（One Interaction Is Worth a Thousand Guesses，NUS + HIT-SZ + ZJU，arXiv:2601.06676，v2 2026-06-20）测 Autonomous 到 Interactive 的增益：

| 模型 | Autonomous | Interactive | 增益 |
| --- | --- | --- | --- |
| Llama-4-Maverick | 54.81 | 65.78 | +10.96 |
| Grok-4.1-Fast | 67.55 | 75.52 | +7.97 |
| Gemini-2.5-Pro | 73.45 | 79.89 | +6.43 |
| Qwen3-235B | 63.96 | 69.67 | +5.71 |
| DeepSeek-V3.2 | 73.35 | 78.64 | +5.29 |
| Claude-Sonnet-4.5 | 75.51 | 80.47 | +4.96 |
| GPT-5.1 | 75.59 | 78.97 | +3.38 |

聚合增益 +6.39，95% CI [5.08, 7.81]。增益与模型能力呈反比：以 autonomous score 作能力代理独立复算 Spearman 为 -0.821；按 headroom 归一化后降至 -0.464，故相当份量是天花板效应。发生了跨模型翻转：DeepSeek-V3.2 交互后 78.64 超过 GPT-5.1 的自主分 75.59。

成本差异极大：DeepSeek 每份报告 +$0.039，Qwen3-235B 为 -$0.006，但 Claude-Sonnet-4.5 是 +$1.233（+125%），Gemini-2.5-Pro 是 +$0.359（+91%）。

构造与目标论文同构：ambiguity injection 把约 800 token 的完整 query 压缩 10% 至 90% 生成欠定 query。

限定：仅 100 instances，只有合并 bootstrap CI，无 per-model CI，故中间四档的细序不可判定，只有 Llama +10.96 与 GPT +3.38 两端显著。User Simulator 固定 GPT-5.1 且 reference-grounded，弱模型等于从更强模型拿特权意图信号，可能系统性放大其增益（论文有三 backbone 稳定性消融、leakage 审计、Limited-Reference 变体三项缓解）。模型阵容停在 2026-06，不含 GPT-5.6 / Claude Sonnet 5 / DeepSeek-V4-Pro，「前沿头部空间只剩 3 至 6 分」的外推性有限。

### 1.4 纯提示与纯算力见顶（`2-1`）

DiscoBench（When Search Agents Should Ask，Tencent Hunyuan + 清华深圳，arXiv:2606.27669，211 samples / 463 ambiguity instances / 11 domains）的 Neutral 到 Guided 消融，10 个有效模型均值（已从 Table 2 逐模型数据独立复算到 0.1）：detection F1 45.3 到 64.9（+19.6），checkpoint pass rate 50.1 到 57.6（+7.6），end-to-end accuracy 28.6 到 33.7（+5.1）。

增益几乎全落在「察觉」而非「做对」。Claude-Opus-4.7 出现 CP 上升而 accuracy 反降（57.0 到 61.6 / 39.8 到 38.9），MiniMax-M2.7 同样反转。原文结论 prompt engineering can partially activate ambiguity-aware behavior, but current models still lack robust and stable clarification ability。

reasoning effort 侧（Doubao-Seed-2.0-Pro medium 到 high，high 已是该模型 API 上限）：均分 45.7 到 54.0，Det. F1 +9.0，Ambi. Rec. 37.2 到 47.3，同样集中在 ambiguity 相关指标；但 even under the high-effort setting, the accuracy remains below 45%。

限定：effort 消融只有单模型、单对照、仅 neutral prompting。reasoning effort 不等于 test-time scaffolding，「纯 scaffolding 见顶」不能从这两条外推——DeepVerifier 与 SAGE-Agent 都是不训练的推理时方法且明显优于强 prompting 基线。本结论是「纯提示与纯算力见顶」，不是「推理时方法整体见顶」。Guided / Neutral 改的是 system prompt 里的通用元指令，q_h / q_o 改的是 user query 内含的任务级维度，层级不同，只能算近似类比。211 样本下 0.9pp 约等于 1.9 题，单模型反转落在噪声内，但 10 模型聚合的不对称稳健。

两个补充数据点（`检索未验证`）：Understanding and Managing Underspecification in LLM Prompts（arXiv:2505.13360v1）测出模型默认只有约 41.1% 能猜中未明说的 requirement；因果识别 benchmark（arXiv:2602.20571）测出 SOTA 模型 79% 能选对 high-level 因果策略，但完整 identification-specification 正确率只剩 34%。即「意识到需要因果分析」相对容易，「把因果论证做对」是另一层墙。

## 二、RSI / 迭代自蒸馏的裁决

原始设想为：闭源模型出 rollout，RL 训开源模型，每步把新权重当老师训下一个。

### 2.1 第二步会塌，且塌的机制有四条独立证据

理论裁决（`检索未验证`）：Self-Improvement in Language Models: The Sharpening Mechanism（arXiv:2412.01951，ICLR 2025）论证 self-improvement 无法创造模型里原本不存在的信息，唯一可解释的机制是 sharpening——把概率质量从自评低分序列挪向自评高分序列。两个边界条件：SFT-based 自蒸馏只在初始模型对高质量解有足够 coverage 时才是 minimax optimal，coverage 不够就停在原地或退化；RLHF-based 的 online exploration 可绕过 coverage 要求。

映射到本课题：若 base model 在 q_o 输入下根本采不到「主动补齐分析维度」的轨迹，纯自蒸馏在理论上就采不出来。

这正是 1.1 测到的东西。自蒸馏的前提是上一轮模型能产出比自身更好的目标规格，而 RubricBench 直接测了这件事并给出否定答案。把该分布当教师，蒸馏目标就是缺失约束的那个分布。

崩塌是悬崖式的（`检索未验证`）：Can Large Reasoning Models Self-Train?（arXiv:2505.21444）用 majority-vote 当伪奖励做无外部标签自训练，初期上涨，延长 RL 后出现 reward hacking，真实性能 sudden and complete performance collapse。数学任务至少有唯一答案作弱锚，deep research 的 analytical checkpoint 连这个弱锚都没有。

同底座 generator 兼 judge 的结构性问题（`检索未验证`）：Spontaneous Reward Hacking in Iterative Self-Refinement（arXiv:2407.04549）在 essay editing 上观察到迭代自我精修自发产生 reward hacking，LM evaluator 与人类判断的偏离在 in-context 内自然扩大，生成器与评估器共享同一底座时优化压力最强。

### 2.2 不塌配方：只 append，永不 replace

Is Model Collapse Inevitable?（arXiv:2404.01413，`检索未验证`）区分两种循环：replace（每代只用上一代模型生成的数据）导致测试误差随代数无界增长，即经典 curse of recursion；accumulate（每代把合成数据叠加在始终保留的真实数据之上）使误差被有界压住，不再随代数发散。

工程判据：迭代自蒸馏能否存活，取决于真实数据（真实检索环境返回的证据、人写的 rubric、含特权提示的真实标注）是否在每一轮都作为固定锚被保留，而不是被上一轮模型的输出替换掉。

### 2.3 需要区分哪一步是「自」

会塌的一步是用上一轮权重生成新 rubric 或新研究维度并当作监督。不塌的一步是用上一轮权重生成 rollout，再按固定外部 rubric 筛选——后者的教师信号来自外部 verifier，属 rejection sampling / RLVR 家族，本来就不是 RSI。

同域存在性证明（`检索未验证`）：AREX（arXiv:2607.21461，BAAI）跑通了 deep research 的递归自改进，但前提是发现难而验证易，「自」只在任务生成侧，奖励侧始终有真实检索环境作非模型自评的锚。与目标论文的 Explorer-Formalizer-Challenger 是同一思路。相关 taxonomy（arXiv:2607.07663）把 bounded self-refinement 与 open-ended RSI 明确切开。

### 2.4 第一步的三个已知坑（`检索未验证`）

heterogeneous distillation problem：MAPD（arXiv:2607.24280）指出闭源 agent 的轨迹风格、协议、tool call 格式与学生自身分布不匹配，纯 SFT 学到表面 pattern 而非搜索策略，且后续 outcome-based RL 难以在其上继续优化。解法是中间插一层轨迹规范化，不要拿 raw trace 直接 SFT。

capacity gap：arXiv:2604.08880 重新检验 teacher-student 能力差过大导致 CoT distillation 失效的结论。用 0.86 分档模型教中小开源模型学「揭示隐藏研究目标」这种分析型元能力，处在高风险区间。

ToS 是硬约束：Anthropic 官方文档（support.claude.com/en/articles/12326764）原文 Our Terms do not allow the use of Outputs to train models that are competitive with Anthropic's own. It is also a violation of our Terms to support a third party's attempt to do the same. 禁止范围是 competitive model，学术用途不构成豁免；「只采集轨迹、别人负责训练」的拆分同样违约。另有主动检测与执法的公开说明（anthropic.com/news/detecting-and-preventing-distillation-attacks）。deep research 任务平均 264.96 次 tool call，长周期大规模采集会形成可识别的调用指纹。

绕开路径：DR Tulu / RLER（arXiv:2511.19399，AI2，ICML 2026）是 8B 全开源、不依赖闭源 rollout 蒸馏的配方，agentic SFT cold-start 加随训练共同演化的 rubric 作奖励，报告在四个 long-form benchmark 上平均超过 Tongyi DR 15.6%。

### 2.5 「特权信息教师 → 无提示学生」的两处实质修正

这两条推翻了本课题最初设想的具体形式，需要单独记录。

修正一，优化目标不该是 q_h 与 q_o 的朴素分差。HiLL（arXiv:2604.00698v1，Snowflake AI Research）定义 hint reliance 衡量正确的 hinted trajectory 有多依赖提示，并证明 hint reliance 越低则 hinted success 向 no-hint success 的迁移越强。推论是单纯最大化 q_h 分数或朴素分差是错的目标，必须显式惩罚「只有拿到提示才做对」的轨迹；正确形式接近 q_o 分数加 reliance 惩罚。但该形式在长程下不可用，见第四节。

修正二，不能用 token-level 蒸馏实现。两条直接打在该设定上（`检索未验证`）：

- Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation（arXiv:2608.04794）的设定与 q_h 到 q_o 几乎同构（teacher 条件在关于答案的特权信息上，student 永远看不到）。结果是 teacher 的 per-token 目标带上 PI bias，蒸馏出一个 flatter, less decisive 的 student，推理能力并未提升，且信号 decoupled from task success。
- The Many Faces of On-Policy Distillation（arXiv:2605.11182）报告 on-policy self-distillation 失败的根因是 instance-specific privileged information 在测试时缺席。q_h 的提示（任务特有的分析维度与约束）正是 instance-specific 而非 generic，属于文献已报告必然失败的那一类。同一病理被 arXiv:2608.01735 命名为 privilege illusion。

另有一条更隐蔽的机制（arXiv:2603.25562）：on-policy distillation 的失败模式之一是 unreliable teacher guidance on student-generated prefixes。当学生在 q_o 下走偏（正是弱点所在的那些 rollout），教师被 condition 在这些 off-target prefix 上时，监督信号的 task-relevant 成分显著衰减。即最需要教师纠正的样本恰好是教师最不可靠的样本。

正面标杆：Π-Distill（arXiv:2602.04942v2）明确针对 hard long-horizon RL，做法是 joint teacher-student objective 而非先训 teacher 再蒸馏。它点出的症结与本课题一致——特权信息破坏标准蒸馏流水线，因为成功行为可观测而推理过程不可观测；rubric 只告诉你 q_h 轨迹达成了哪些 checkpoint，不告诉你如何从模糊提问推断出这些维度。

## 三、「何时反问」的判据与校准问题

### 3.1 决策论判据已存在且可落地

两套框架。VoI（Value of Information: A Framework for Human-Agent Communication，Cambridge + MIT，arXiv:2601.06407v1）给出 V(b) = max_a EU(a|b)、V_post(b,q) = Σ_y p(y|q,b)V(b_y)、VoI(q) = V_post - V(b)、NetVoI(q) = VoI(q) - c，停止规则原文 If max_q NetVoI(q) ≤ 0 ... the agent terminates the dialogue and commits to the best action under its current belief。

SAGE-Agent（Structured Uncertainty guided Clarification for LLM Agents，Adobe Research + UMD，arXiv:2511.08798，ACL Findings 2026）给出 q*(t) = argmax_q[EVPI(q,B(t)) - Cost(q,t)] 与双阈值：max_c π_c(t) ≥ τ_exec 直接执行；max_q[EVPI - Cost] < α·max_c π_c(t) 停止提问（λ=0.5, α=0.1, ε=1e-4）。附 EVPI 非负、次模、收敛证明与 Finite Termination 定理，回合上界 T ≤ EVPI_initial/(αρ+γ)。

ClarifyBench GPT-4o Ambiguous 上对全部四个基线同时 coverage 更高且提问更少，是严格 Pareto dominance：Coverage 59.73 对 ReAct+ask_question 的 42.88（相对 +39%）、对 Domain-aware ReAct 的 55.70（+7%）；Avg#Q 1.39 对 2.07 至 3.42（1.49 至 2.46 倍）；TMR 86.02 对 70.41。λ 由 0 到 0.5 使提问数降 18.1% 至 26.6% 而 Coverage / TMR / PMR 偏移小于 3%，原文结论 confirming penalized questions were redundant。

需要打折：VoI 论文 abstract 的 consistently 与 in high-cost settings 对其自身 Table 1 有水分——真实实验量是 3 个 task × 2 档 cost × 2 模型加一个 5 点 sweep，不存在「4 域 × 5 cost = 20 条件」全网格；+1.36 出现在 c=0.05 的中间档，真正的高成本档 c=0.10 与 c=0.20 恰是 VoI 落败的两次（-0.90、-0.96）。SAGE 的 ClarifyBench 是同篇自建的 simulated environment，Coverage SD 为 ±22 至 34、无显著性检验、3 runs 取 best，故 +7% 下界很可能不显著；1.5 至 2.7 倍主要由 GPT-4o 驱动，Qwen2.5-14B 上多处低于 1.5 倍。

### 3.2 置信度阈值不可用

VoI 论文 §6.1 有独立小标题 Confidence thresholding is effective but brittle. 原文：the optimal τ is highly sensitive and must be manually selected for each task and cost combination, making it impractical for real-world deployment。Table 1（Gemini-2.5-Flash，grid search over 9 values）最佳 baseline 随成本换了四种配置：cost 0.01 Confidence τ=0.9（8.30），0.02 同（6.88），0.05 Round τ=5（3.65），0.10 Confidence τ=0.5（2.28），0.20 No Question（0）。

旁证同向（`检索未验证`）：arXiv:2603.26233、arXiv:2601.07264、arXiv:2604.08588 均指向 LLM 内部置信度校准差，不足以单独作 ask/act 判据。

限定：该 brittle 判词出自提出竞争方法的作者对自建 baseline 的评价，属自利框定，且 VoI 本身在 cost 0.10 / 0.20 反而输给最佳调参 baseline。该论文实验域不含 deep research。

### 3.3 判据只解决「何时问」，不解决「问什么」

这是最明确的空白，且是两篇作者自陈的 Limitations。VoI 原文：Our work focuses on the core decision of when to communicate, rather than what questions to generate，并承认 Extending this framework to fully open-ended dialogue is an important next step。

可算性前提逐条有原文支撑：Eq.1 是有限求和且 Θ 为给定候选集；U(θ,a) 是算法输入；答案空间被强制收窄且理由就是可算——To ensure computational tractability in Eq. 3, we constrain the agent to ask closed-ended questions ... thereby defining a finite answer space Y。四个实验域的动作集全是小规模预枚举（100 种动物 / 15 种疾病 / 3 个航班 / 对 ground-truth product 打分）。SAGE 的精确性红利同样来自 tool schema 的可枚举有限参数域（有限域 ≤20 值时完全枚举，连续域退化为常数 ε）。

含义：要把判据用到开放式研究目标发现上，必须先额外构造一个离散化的「研究维度域 / 候选比较对象集 / 因果与相关性主张集」。这一步两篇论文都没做，而目标论文的 DAG（E/A 节点加 checkpoint）是这个离散化的现成候选。

需软化一点：SAGE 的 GenQ(H) 表明框架本身允许在线生成候选问题，严格说依赖的是「有限」而非「必须预定义」。因此空白强度应表述为「这两个框架未覆盖，且现有 coding / search agent 澄清工作亦未给出开放式目标发现的判据」，而非全域文献调研结论。

### 3.4 校准误差进入判据是二次衰减，但前提 LLM 不满足

两条独立显式界。CDL 界（arXiv:2404.13503）：在所有 payoff 归一化到 [0,1] 的下游决策任务上，ECE² ≤ CDL ≤ 2·ECE，两端在常数因子内 tight。belief 的 ECE 为 ε 时任意 best-response 决策的最坏收益损失不超过 2ε；当预测值方差大时 ECE 会二次地高估实际决策损失（ECE = 1/√T 而 CDL 仅 Θ(1/T)）。

prior 误差界（arXiv:2210.03905）：估计先验误差为 O_p(r_n) 时，相对于知道真先验的 Bayesian oracle 的选择 regret 为 O_p(r_n²)，参数情形给出 O_p(n^-1) 而非通常的 O_p(n^-1/2)。

机制是排序反转自限：误排只发生在价值差已经很小的阈值附近，因此 rank preservation 并非判据保值的必要条件。

但成立条件恰好是 LLM 不满足的那一条：要求误差是随样本消失的估计噪声，且先验族包含真值。误设定下保护实测崩塌——用误设定的正态先验时 regret 完全不收敛到零，误选比例与幅度不收敛甚至随实验数增加而上升。LLM 的误校准是持续性、有方向的偏差，r_n 不趋零。因此不能主张二次衰减会替判据兜住误差。

### 3.5 belief 估计端的三条坏消息

候选基数增大时置信度系统性失真，且模型越大越严重。MACE（arXiv:2602.07842，12,000 题、6 领域、每题 1/2/4/6 个正确答案、15 种校准方法、4 个模型家族 7B 至 72B）上 LLaMA-3.1-70B 的 accuracy 从 1 答案的 48.0 升到 6 答案的 61.7，而 consistency-based confidence 从 51.3 掉到 35.9、Sem Entropy 从 60.8 掉到 45.2、Verb-Topk 从 81.9 掉到 56.9，方向相反。规模效应：70B 从 1 到 6 答案的 confidence 下降 0.154 而 8B 只下降 0.056（约 3 倍）。机制是更广的 knowledge coverage 产生更多样的等价正确答案、稀释 response consistency。用 frontier 模型跑 belief 估计不会缓解，反而放大。

三类置信度信号绝对校准全线很差，且「哪类更校准」是测量 protocol 的产物。固定模型自身答案与 correctness label、只把 token 打分挪到两个都合法的 prompt context 之间，就在 4 QA 数据集 × 3 个 7 至 8B Instruct 模型的 12 个设定中翻转赢家 4/12（ECE）与 9/12（AUROC）；macro ECE 为 verbalized 0.426、token 三个 context 0.257 至 0.358（arXiv:2605.27752）。AUROC 对保序变换不变，故 0.604 到 0.762 的位移是重排而非缩放，Platt / temperature（严格单调，保 AUROC）与 isotonic（弱单调）都无法补回。

完美校准也不够。即使重校准到完全校准，grouping loss 仍留下 regret：下界 U_Δ[GL(p) - V_min(p)]₊，上界 ½U_Δ(√(GL(p)+(c(p)-t*)²) - |c(p)-t*|) ≤ ½U_Δ√GL(p)，且 V_min(p) 在 c(p) = t* 处恰为 0。即校准后概率正落在决策阈值上时任何 grouping loss 都必然产生 regret，且按 √GL 放大而非线性（arXiv:2503.18025）。

### 3.6 验收指标不能用 ECE

实测：ECE、MCE、RMSCE、CL 四个校准误差指标对 5 种重校准方法带来的效用增益相关性都极差（r² ≤ 0.1），而 decision-aware 的估计校准 regret 与效用增益相关到 r² = 0.88（6 个预训练模型 × 14 个真实数据集 × 11 个 t*）。

理论上更强：CDL 与 smooth calibration error / distance to calibration 并非多项式相关——存在序列使 distCal ≤ 1/√T 可任意小而 CDL = Ω(1)，因此 online 校准里 Õ(1/√T) 的 distCal 保证对决策损失给不出任何界。反面出口：ECE 与 K₂ 仍与 CDL 多项式相关，被否掉的只是平滑类指标。

### 3.7 域大小不是瓶颈（`2-1`）

CDL 等于对所有 payoff 有界于 [0,1]、动作空间任意（可无限）的决策任务取 best-response swap regret 的上确界，且存在多项式时间在线算法把期望 CDL 压到 O(log T/√T)，该保证不含 |A| 依赖（改进了 Roth-Shi 2024 的 O(|A|√(log T/T))），并绕过 Qiao-Valiant 对 ECE 的 Ω(T^-0.472) 下界。

含义：把研究维度域从 20 扩到 40（对应 DAG 平均 20.85 节点）不会以规模项恶化这类保证的率。可算性与精确 EVPI 枚举的组合爆炸是实现问题，不是信息论障碍。

### 3.8 未找到 VoI 自身的误差传播界

本轮 6 条候选界被逐一否决（`被否`，1-2 与 0-3×5），包括 EIG 排序不保序的最小反例、KL 型 EIG 误差界与 robust surrogate 的 O(ε²) 二阶界、有限样本误差与误校准合并为同一 ambiguity radius、判据 argmax 必不保序、阈值穿越型闭式 regret、regret 随 α·m 线性放大。

所有可用的定量界都来自三个邻域：binary threshold decision、top-m selection、Bayesian experimental design。跨域迁移到多维 VoI argmax 的风险由使用者承担。

另有一条同向的结构性结论（arXiv:2506.07805）：纯 informativeness 判据在 belief 误设定下不鲁棒，鲁棒性需要 representativeness 与 de-amplification 两个性质而 EIG 对两者都没有对应项；且误差走两条路径复合（模型同时用于拟合推断与选数据），映射到本课题即 LLM 的 b(θ) 既进 EVPI 计算又进后续 belief update。实测出现非单调与反序。

### 3.9 问题 A 的可操作配方（`推断`）

以下从上述界推导，非任何单篇文献的结论。

域大小方面，决策端把 DAG 20 至 40 节点保留在结构里是安全的，但不要在全部维度上一次性做 argmax；每步只在 top-k（k 约 5 至 8）候选上比较 EVPI，把平手邻域的暴露面压小。

belief 估计方面，不要用 verbalized confidence（macro ECE 0.426，最差一档）；token-likelihood 略好但绝对值仍差；必须把 answer、context、evaluation protocol 三者显式写死并在所有比较中保持一致，否则结论是 protocol 的产物；不要指望换 frontier 模型缓解。

后处理方面，单调重校准修不了重排，只能修 level，因此不能作为唯一防线。

验收方面，门禁必须是 decision-aware regret 估计量，禁止用 ECE 阈值当准入，更禁止用 smoothECE 或 distCal。

判据形态方面，把「精确 argmax 加 NetVoI ≤ 0 停止」改成「带 margin 的 argmax 加保守停止加硬预算上限」：只有 top-1 与 top-2 的 NetVoI 差距超过由 belief 误差预算导出的 margin 时才信任排序，否则退回固定顺序或轮询。理由是排序反转在平手处的价值损失本已很小，而 grouping loss 恰在阈值处最具破坏性，两者共同指向「平手处别让判据做决定」。NetVoI ≤ 0 是最脆弱的一环，必须配硬预算兜底。

总判断：判据不会被无条件吞掉，但净收益的存亡取决于能否把 belief 误差从持续方向性偏差压成小且方向随机的噪声。若做不到，应放弃精确 EVPI 数值，只用它做粗粒度排序过滤。

### 3.10 本轮未覆盖

有无人把 latent intent 空间用任务分解图、DAG、checklist、aspect 集合离散化后再算 VoI / EVPI / entropy-based disambiguation，其 belief 如何估、做了什么校准处理，本轮零覆盖。属「未找到证据」而非「证据表明无人做过」。

## 四、hint reliance 惩罚与长程 agent

### 4.1 没有人做过

同族四篇方法的实测 horizon 全部是单轮静态问答，含真实工具调用的多轮 agent 数据点为 0，相对目标场景差约两个数量级。

| 方法 | 实测设定 | horizon |
| --- | --- | --- |
| HiLL | OpenR1-Math-220k 15k prompts；AIME24/25、AMC23、MATH-500、Minerva、OlympiadBench、GPQA-diamond、MMLU-Pro | 单轮数学，全文 tool / agent / multi-turn / environment / episode 各出现 0 次 |
| HIPPO | DeepScaleR 与 MedQA 各约 5,000 条 | 单轮 QA，max response 2048 / max prompt 1024，tool 一词只出现在参考文献标题 |
| SAGE | 64k prompts，6 个数学 benchmark | 单轮数学，无工具 |
| RuscaRL | HealthBench、LLMEval-Med、WritingBench | 族内最长但仍单轮，max gen 4096（非写作）/ 16000（写作） |

HiLL 硬配置：max prompt 2048、max response 8192、temperature 0.6、top-p 0.95；reasoner 训 500 steps、batch 128、G=8 rollouts/prompt、lr 1e-6；hinter 每个 all-incorrect 问题生成 M=4 候选，max prompt/response 10240/1024，transfer temperature T=0.3，R_fail=-0.2；reasoner 为 Llama-3.2-3B-Instruct 与 Qwen2.5-7B-Instruct，hinter 由 Qwen3-4B-Instruct 初始化。

SAGE 增益量级很小：平均 +2.0（Llama-3.2-3B）、+1.2（Qwen2.5-7B）、+1.3（Qwen3-4B）。另有一个值得注意的三分消融：Fixed privileged hints 对更强外部模型 hint 对 online self-hinting，结论是 online self-hinting 最优，即静态特权提示会随 learner 漂移而失效。

RuscaRL 存在退化行：HealthBench-500 +14.9 / WritingBench +17.x，但消融配置下有 -3.1 / -11.x，说明 rubric 指引配方并非单调有效；且检索文本中未见任何「评测时撤掉 rubric 后的迁移率」量化。

### 4.2 HiLL 的迁移保证在长程下必然 vacuous

定理里的量与实现里的量不是同一个。Proposition 1 用未归一化的整轨迹 log-ratio ρ_c，代码（hill_ray_trainer.py:779-782）除以 token 数。论文未讨论任何 horizon 依赖，也未给出任何方差界或曲线。

保证形式 p ≥ p_h·exp(-ρ_c) 在长程下塌掉：

| 轨迹长度 | 界的值 |
| --- | --- |
| 8,192 token（HiLL 实测） | 2.6e-36 |
| 24,576 token（参考实现 max_response_length） | 1.9e-107 |

估计样本只有 |C| ≤ G = 8 条正确轨迹。hint reliance 定义为 ρ_c(q,h) = log(p_h/p) + D_KL(P_h(·|r=1) ‖ P(·|r=1))，实际估计量 ρ̂_c = (1/|C|)Σ_{τ∈C} ρ(τ;q,h)/|τ| 只在 hinted 且正确的轨迹上取平均；长度归一化只压住量纲，压不住样本稀缺带来的估计崩塌。

相关但需区分的一条：HIPPO 附录 A.4 证明将 hint 依赖度写成 autoregressive 分解后，当 token-level log-ratio 有共同正方差下界 σ² > 0 且相关性非负时 Var[r*] ≥ T·σ²，即至少随轨迹长度线性增长。该界本身经一手全文逐字核验无误，但 HiLL 实际用的是长度归一化版本，故不是同一个量。两者结论一致（不可直接估），机制不同。HIPPO 给出的替代模板是用 Pinsker 不等式加 TV 变分表示把 KL 目标 reduce 成 pairwise comparison reward，方差与 T 无关。

### 4.3 HIPPO 的 hint 方向与同族相反

HIPPO 的 hinted rollout 是被故意污染的负向 anchor，被优化的 policy 全程不接收 hint。因此它结构上不存在「训练给提示、评测撤提示」的迁移，无法提供任何 hinted→unhinted 迁移率数据点。代码在 github.com/Infinite-set/HIPPO。

### 4.4 长程 agent RL 的 horizon 标尺

| 来源 | horizon |
| --- | --- |
| GiGPO ALFWorld | ≤ 50 env steps，>20k tokens/episode |
| GiGPO WebShop | ≤ 15 |
| GiGPO Search-QA | max turn 4，实测均值 0.9 至 1.6 tool call |
| TRACE | 60 train / 80 eval tool turns |
| 目标场景 | 264.96 tool calls |

GiGPO 对 hint / privileged / teacher-student / asymmetric / distill 五个关键词全文 0 命中——它提供步数标尺，不提供迁移证据。Search-QA 的实测均值仅 0.9 至 1.6，说明现有「多轮检索 agent」benchmark 的实际 horizon 比标称低得多。

### 4.5 step-level 估计量的样本量由状态复现率支配

step-level 反事实或 leave-one-out 式估计量的可用样本量不随 horizon 自动增长，而由状态复现率支配。关键补充：早期的样本充裕度部分由退化行为贡献（invalid actions、repetitive loops）；策略变强后每状态可配样本数向 N 收敛而非向 T 增长。

即这是训练过程中会自己恶化的量，一次测量不能定。初期的好数字不可当准。

GiGPO 的 anchor state grouping 是 GRPO 无 critic 约束下现成可用的第二层 group baseline，开销可忽略、无额外 rollout、无显存增量；但其分组 key 是可观测环境状态而非特权信息，因此不是非对称 baseline 变体，也不提供任何 hint 机制。

削弱证据（`检索未验证`）：ECPO（arXiv:2606.05885）报告 GiGPO 的 divergent anchor 占比随训练从 9% 涨到 28%、final reward σ 为 0.746（ECPO 校准后 0.555）、over-rewards rare lucky actions、late-stage oscillation，校准后在 ALFWorld / WebShop 以 Qwen2.5-1.5B 超出 +5.2 / +7.3 success points。另有 ProGPO（arXiv:2607.04242）指出 33.7% 的 WebShop 步与 44.5% 的 ALFWorld 步没有 peer 信号。

结构性推论（`推断`）：ALFWorld 类环境状态空间封闭且小，不同轨迹反复经过同一状态，anchor grouping 才有得可分；deep research 的状态是「已检索到的证据集合」，随内容组合爆炸，状态复现率先天更低。因此任何依赖状态复现的 credit assignment 在本场景先天不利。不依赖状态复现的机制包括：冻结评估器读完整轨迹、特权信息进 optimization 而非分组、同一 prompt 的 sibling rollout 成功失败对比。

### 4.6 CRAFT 的识别性缺口（`2-1` 与 `3-0`）

CRAFT 是「GRPO 无 critic 加特权信息走 credit 通道」的已发表实例：复用 GRPO 本就要采的 G-1 条 sibling rollout，用 teacher-student log-prob gap 做 self-normalised importance weighting，得到有符号 per-token credit，detach 后按 REINFORCE 进 loss。

自承缺口即长程下不可估的直接证据：group-level estimand 等同于 per-trajectory counterfactual 依赖 within-group exchangeability 假设 (E)，而 (E) 只在 t=0 精确成立、之后 smoothly degrades，且该 gap 完全未被量化。

其方差界 Var(ĈTI_t) ≤ R_max²ρ̄_t²/(G-1) 是 horizon-free 的 O(1/G)，即根本没建模 horizon 效应。CRAFT 全文不报任何 per-episode 步数、tool call 数或轨迹 token 长度，只用形容词指代 horizon。

### 4.7 文献里不存在 horizon-dependent 的方差界

唯一试图给出「方差随 horizon 增长」定量律的工作（Drowning in Routine，arXiv:2606.22164）没有任何真实 LLM agent 实验，策略仅 30k 参数、2 层、d_model=32，其定量结论本轮未通过验证（`被否`）。通过验证的只有作者自承的三条局限与两条 critic-free 补救建议，且补救前提（检出 routine turn）本身尚无方法。

其机制描述通过：decision density 低时 routine turns add gradient variance to trajectory-level estimators such as GRPO without adding expected signal。定量律未通过。

同向表述（TRACE）：outcome rewards become sparse and high-variance as trajectories grow to tens or hundreds of tool calls，量纲与目标场景匹配。

### 4.8 hinted→unhinted 迁移零公开数据点（`2-1`）

最接近的证据是 ECHO 把 privileged oracle 当上界诊断，测得 40.9 个百分点差距，且两者最优行为分布在质上反转。这是「特权策略动作分布不可被无特权 student 直接模仿」的带数字机制证据，但不是迁移率测量。

ECHO 另给出两条否定结果：belief-agnostic policies 的误差可随 horizon 指数复合；aggregate trajectory returns 可以无法辨识 per-turn Bayesian advantage。第二条是 identifiability 层面的否定——轨迹级 return 原则上不足以辨识逐步贡献，必须额外注入 posterior-sensitive 或 turn-level 信号。

curriculum 侧（`检索未验证`）：DAHS + BHA（arXiv:2604.07747）给出 Backward Hint Annealing，按 difficulty bucket 退火 hint exposure 并用 per-question hint dropout 在全程保留 no-hint 更新，失败模式命名为 distribution sharpening。ADHint（arXiv:2512.13095）给出 per-sample 自适应 hint-ratio schedule 加对 hinted/unhinted 轨迹差异化的组内 advantage 估计，并点出前人两个失效模式：unstable learning 与 excessive imitation of off-policy hinted text。SEELE（arXiv:2509.06923v1）调节 hint length 并用 IRT 拟合选每实例最优提示长度。三者均为单轮数学或 reasoning 设置。

元评价（TUM 2026 thesis）：recent work has shown that gradually dropping privileged information can aid transfer, but these methods lack theoretical grounding and systematic design principles。即渐进撤除有效已被多次观察到，但缺理论保证与设计原则。

### 4.9 TRACE 是当前最贴合的配方

TRACE 给出目前唯一在 60/80 tool-turn 量级配置、真实检索环境上跑的 critic-free 非对称特权配方：gold answer 只在训练期可得，且只进 frozen reference 的打分通道而不进 policy context；turn credit 由 log-ratio TD 差得到，与 GRPO outcome advantage 线性混合。

其他 GRPO 侧可用配方（`检索未验证`）：

- ActGuide-RL（arXiv:2605.12004）把 guided 与 unguided rollouts 放进同一个 group，advantage 仍由同组 unguided 轨迹定标，评测期撤掉提示不需要额外 annealing schedule。
- OC-GRPO（arXiv:2607.19313）在含 privileged guidance 的 prompt 下采 rollout，用 importance-corrected objective 把梯度掰回不含 guidance 的原始 prompt 目标，相比 vanilla GRPO 平均 +3.9% 绝对 / +13.8% 相对，开销 negligible；但实验域是单轮数学。
- SGCD（arXiv:2606.12634，AWS）标题即锁定 long-horizon tool-use：dynamic sampling 产出成功失败混合的 sibling rollouts，外部 LLM 对比总结成 training-only credit reference，再用 detached teacher/student divergence 去 reshape GRPO advantages；作者强调是用 distillation 做 bounded credit weighting 而非与 policy gradient 竞争的 actor loss。
- CriticSearch（arXiv:2511.12159v1）在真实 tool-calling search agent 上用冻结的非对称 critique LLM 以完整轨迹加 gold answer 的特权信息回溯评估每轮，转成 dense per-turn 信号；结构上是 asymmetric-critic-without-a-critic-network，与 GRPO 兼容。
- PACT（arXiv:2606.16215）rollout 保持 prompt-only，expert traces 只进 optimization，等价于不给 rollout 提示因此不存在 hint reliance。
- POPE（arXiv:2601.18779）oracle solution 仅引导 on-policy exploration，never 当训练 target。

负面实测（`检索未验证`）：Self-Distilled Agentic RL（arXiv:2605.15155v1）报告把 on-policy self-distillation 搬到多轮 agent proves problematic: compounding multi-turn instability destabilizes supervision, while skill-conditioned privileged guidance requires asymmetric treatment，并称自己避开了 the instability of naive GRPO+OPSD，WebShop-Acc +10.2%。即直接把特权提示蒸馏拧到 GRPO 上在多轮 agent 里不稳定有公开数据点，修法是对特权分支做显式非对称处理。

### 4.10 问题 B 的判断

长程下 hint-reliance 惩罚按 HiLL 原定义不可直接估。可落地替代按优先级：TRACE 式特权-reward 通道，其次 CRAFT 式 privileged group-relative advantage，其次 GiGPO 式 anchor-state 二层 baseline（需自定义 state 抽象且需先量复现率），reliance 仅作监控量，最后自建 hint annealing ablation。asymmetric critic 与 GRPO 无 critic 约束冲突且本轮无支持证据。

### 4.11 本轮未覆盖

机器人学 sim-to-real 的 privileged teacher→student 成熟配方（asymmetric actor-critic、Learning by Cheating、RMA、DAgger-style privileged distillation、asymmetric PPO 的已知失效条件）零覆盖，属「未找到可核验来源」。

检索阶段曾命中以下线索但均未进入存活 claim，引用前必须自行核实：Informed Asymmetric Actor-Critic（arXiv:2509.26000，称任意特权信号都给出无偏 policy gradient，且特权信息越多不是单调越好）、To Distill or Decide?（arXiv:2510.03207，NeurIPS 2025，称 the optimal latent policy is not always the best latent policy to distill）、Provable POMDP with Privileged Information（arXiv:2412.00985，NeurIPS 2024，称多项式复杂度仅在 deterministic filter condition 下成立）、Distilling Realizable Students from Unrealizable Teachers（arXiv:2505.09546）。

## 五、目标论文代码仓库核查

核查日期 2026-08-19，方法为 GitHub API 读取 tree 与文件内容（网页与 raw 端点在本环境超时，API 可用）。

仓库为 github.com/chr6192/TaskEvolving，Apache-2.0，默认分支 `0720`（非 main/master），创建 2026-08-03，最后推送 2026-08-10，0 star，2,392 KB，110 个 tree 条目（未截断）。子模块三个，均钉死版本：verl `v0.4.1`（8d9e350e）、OpenJudge `v0.2.2`（33db7c4a）、natural-questions（fb26a307）。

### 5.1 训练栈已实现，且形态与本调研的建议一致

仓库描述为 Automatic Task Evolution for Deep Research, and the Agentic RL Training Stack to Train on It。README 原文：the reward function uses an OpenJudge rubric grader to score a model's multi-turn tool-use trajectory checkpoint-by-checkpoint, aggregates the scores by their DAG-derived weights, and feeds the result to the RL algorithm (PPO/GRPO, etc.) as a process reward signal。

构件：`dr_ray_trainer.py`（扩展 verl 的 RayPPOTrainer）、`dr_completion_callback.py`（多轮 tool call 解析与派发）、`dr_chat_scheduler_no_thinking.py`、`reward/openjudge_reward_function.py`（compute_score_async）、`reward/openjudge_reward_manager.py`（桥接 DataProto）、`run_qwen3_async_mt_dr.sh`（GRPO 加 multi-turn 加 async vLLM rollout）。

即 rubric/checkpoint 级 outcome-anchored process reward 这一层不需要自建，Apache-2.0 可直接用。

### 5.2 500 个任务的 DAG/rubric 数据不在仓库里

完整 tree 中 data 类文件仅 6 个 yaml/txt config（4 个 tool_config、2 个 synthesis config）加 requirements.txt，无任何任务数据。训练脚本的 `data.train_files="$TRAIN_DATA"` 是外部环境变量。

论文摘要称 Our data, implementation, and results are publicly available，实际只放了 implementation。锚数据需自行重跑生成，量级参考论文自身统计：平均每任务 14.19 轮演化、264.96 次 tool call（最少 32、最多 1854），synthesis config 的 `default_model_name` 为 gpt-5.5、`max_request_input_tokens` 为 240000、`multi_turn.max_turns` 为 200。

合规提示：用闭源模型生成 rubric 属评测用途风险较低，但若这些 rubric 直接变成 RL 奖励信号即进入训练回路，落回 2.4 节的 ToS 灰区。可考虑改用许可允许再训练的开放权重模型做 Formalizer。

### 5.3 训练 rollout 的 horizon 与任务本身差 20 倍

`run_qwen3_async_mt_dr.sh` 的 `multi_turn.max_turns=10`、`max_prompt_length=8192`、`max_response_length=24576`、`train_batch_size=16`、`ppo_mini_batch_size=16`、`log_prob_micro_batch_size_per_gpu=1`、`algorithm.use_kl_in_reward=False`、`data.truncation='error'`；而 synthesis 侧 config 为 `max_turns: 200`。

`truncation='error'` 意味着上下文超限会硬失败而非静默截断，这一点有利于尽早发现问题。但该 20 倍差距说明作者的训练栈未在 benchmark 原生 horizon 上跑过，与第四节的长程结论直接相关。

### 5.4 q_o 生成可复现，但没有任何 hint 机制

fuzzify 三件套在仓库内：`fuzzify_query_prompt.py`、`fuzzify_query_prompt_en.py`、`fuzzify_query_validation.py`（6,602 字节，程序化自检），config 有 `fuzzify_max_attempts: 3`。因此 (q_h, q_o) 配对可批量生成，不受数据缺失影响。

fuzzify 的硬门槛包括 `len(q_o)/len(q_h) ≤ 0.70`、逗号段 ≤3、句子 ≤3、问号 ≤2、禁止流程词、以及「读出来像不像真人提问」的自检。两个含义：压缩率是被强制的而非自然产生，因此可作为 hint annealing 的确定性旋钮（放宽到 0.85 或收紧到 0.50 即得难度梯度）；论文的 0.16 gap 是在 ≤0.70 这一档测出的，用不同压缩率造的数据不可与论文数字直接比较。

reward 配置为 `rubrics_only: true`、`grader_weights: {rubrics_based_trajectory_performance: 0.2}`、judge 为 qwen3.7-max、temperature 0.0、`language: "cn"`、`max_concurrency: 32`。

训练栈读单一 query 形式，无 q_h/q_o 差分、无 reliance 惩罚、无逐步撤提示。即作者手上有全部原料（同一套 rubric、现成 fuzzify、跑通的 RL 栈）却没做这一步，与 2.5 节的空白判断一致。

语言不一致需注意：`complexity_exploration.language` 默认 `cn`、reward function `language: "cn"`，而 seed 来自英文 NQ-Open；仓库同时提供 cn 与 en 两版 prompt。复现时需对齐，否则 rubric 语言与轨迹语言错配会影响 judge 打分。

另注：`reward/openjudge_reward_manager_readme.md` 内引用的是 `open-compass/OpenJudge`，而主 README 与 tree gitlink 指向 `agentscope-ai/OpenJudge`，两个不同 org。以主 README 与 gitlink 为准。

## 六、与本项目（verifiable_self_evolution）的关系

本节是本文档对 `README.md` 与 `PROGRESS_V0_1_1.md` 所述设计的外部文献支撑与风险提示，不修改任何既有设计或门禁。

### 6.1 本项目的核心设计与文献结论一致

本项目的 closed teacher 两项受限角色（仅产生 generation zero 的 bootstrap candidates、仅修复机器识别的 hard state）、teacher output is never accepted as truth、teacher 不可见 held-out/OOD、teacher 不参与 promotion 判定，恰好对应 2.3 节的划分：会塌的是「用上一轮权重生成新的目标规格并当监督」，不塌的是「生成候选后由外部 verifier 筛选」。

promotion gate 不消费 training results、teacher judgments 或可变 task list，对应 2.1 节 Spontaneous Reward Hacking 的结论——generator 与 evaluator 共享底座时优化压力最强。本项目用 frozen promotion evaluator 加 preregistered gate 切断了该通路。

success library 与 counterexample library 始终保留、QLoRA 从 frozen replay buckets 导出，对应 2.2 节的 accumulate 而非 replace。这条是 model collapse 文献给出的唯一「不塌配方」，本项目的 export-training-data 三桶设计（teacher-anchor、verified-success、corrected-counterexample）与之相符。

### 6.2 对 recursive 配置的风险提示

`configs/paper_rediscovery_recursive_v1.json` 的三代递归设置面临 2.1 节的两条风险：sharpening 机制意味着若 base model 对目标行为的 coverage 不足，递归只会放大已有行为；无外部标签自训练的崩塌是悬崖式而非渐进的，依赖 early stopping 躲开不可靠。

本项目的缓解是每代都有 hard verifier 与 frozen evaluator，属于 RLVR 家族而非纯 RSI，因此上述风险不直接适用。但需要注意的是 2.1 节 RubricBench 的结论针对的正是「模型自主归纳评价标准」这一能力——若未来任何一代的 verifier 或 rubric 由模型生成而非人工/解析给定，该代即退化为纯 RSI，崩塌风险恢复。当前 README 要求 analytic or independently implemented reference solver，满足这一条。

### 6.3 若本项目未来引入 privileged hint

第四节的结论直接适用，简述为三条：hint reliance 类惩罚在长程下不可直接估（4.2）；特权信息应只走 reward/value 通道而不进 policy context（4.9 的 TRACE 式）；token-level KL 蒸馏在特权设定下会产出 flatter、less decisive 的 student 且信号与 task success 脱耦（2.5）。

### 6.4 未在本项目范围内的部分

第三节（VoI/EVPI 判据与校准）针对的是 agent 向用户反问的时机决策，本项目当前无交互澄清通道，故该节暂不适用。若未来引入，3.9 节的配方与 3.6 节的验收禁令（不得用 ECE 当门禁）为可直接引用的约束。

## 七、待核实与空白清单

以下四项在三轮检索中均未获得可核验证据，若要依赖其中任何一条，需另起定向检索。

model collapse 与 sharpening 的定量条件：需要多少比例的人类或外部数据、需要多强的 verifier 才不塌，本轮理论侧文献（arXiv:2412.01951、2404.01413、2505.21444、2407.04549、2504.13837）全部为 `检索未验证` 状态，未进入 3 票验证。

把 q_h 与 q_o 在同一 DAG/rubric 下的分差直接当优化目标：教师侧 headroom 已被两处独立量化（+26pp 与 +6.39 分），但把该 headroom 蒸馏进无提示学生这一步没有公开做法或失败案例。

开放式研究维度域的离散化：DAG 是否可充当 VoI/EVPI 所需的有限域，以及此时 belief 校准误差是否吞掉判据收益，本轮 3.10 节零覆盖。

机器人学 sim-to-real 的 privileged teacher→student 配方迁移性：4.11 节零覆盖。

## 八、方法与来源

三轮检索的规模与产出：

```yaml
round_1_defect_and_rsi:
  agents: 110
  sources_fetched: 28
  claims_extracted: 138
  claims_verified: 25
  confirmed: 9
  killed: 16
  findings_after_synthesis: 8
  coverage: 方向1与2充分；方向3/4/5的材料检索到但未进验证队列
round_2_calibration_and_voi:
  agents: 109
  sources_fetched: 27
  claims_extracted: 134
  claims_verified: 25
  confirmed: 17
  killed: 8
  findings_after_synthesis: 13
  coverage: 问题A充分；问题B零覆盖
round_3_hint_reliance_long_horizon:
  agents: 106
  sources_fetched: 24
  claims_extracted: 120
  claims_verified: 25
  confirmed: 14
  killed: 11
  findings_after_synthesis: 14
  coverage: 子问1/2/3充分；子问4零覆盖
```

验证协议为每条 claim 起 3 个独立 agent 投票，需 2/3 判定推翻才淘汰。三轮合计 392 条 claim 抽取、75 条进入验证、40 条存活。验证预算是主要约束：第一轮 5 个方向摊薄预算导致方向 3/4/5 的材料未获验证，后两轮收窄到 1 至 2 个问题后覆盖率改善。

主要来源按主题分组。目标论文与其实现：arXiv:2608.02163、github.com/chr6192/TaskEvolving。

缺陷定位与评测：arXiv:2603.01562v1（RubricBench）、arXiv:2601.06676（IDRBench）、arXiv:2606.27669（DiscoBench）、arXiv:2505.13360v1、arXiv:2602.20571。

澄清判据：arXiv:2601.06407（VoI）、arXiv:2511.08798（SAGE-Agent）、arXiv:2605.07937。

校准与决策损失：arXiv:2404.13503、arXiv:2210.03905、arXiv:2503.18025、arXiv:2605.27752、arXiv:2602.07842（MACE）、arXiv:2203.09852、arXiv:2506.07805、arXiv:2504.15582。

RSI 与自蒸馏：arXiv:2412.01951、arXiv:2404.01413、arXiv:2505.21444、arXiv:2407.04549、arXiv:2504.13837、arXiv:2607.21461（AREX）、arXiv:2601.15808（DeepVerifier）。

特权信息与蒸馏：arXiv:2604.00698v1（HiLL）、arXiv:2602.03143（SAGE）、arXiv:2606.29481（HIPPO）、arXiv:2508.16949（RuscaRL）、arXiv:2602.04942v2（Π-Distill）、arXiv:2605.11182、arXiv:2608.04794、arXiv:2603.25562、arXiv:2608.01735。

长程 credit assignment：arXiv:2607.13988（TRACE）、arXiv:2606.29476（CRAFT）、arXiv:2505.10978（GiGPO）、arXiv:2606.22164、arXiv:2606.29745（ECHO）、arXiv:2606.05885（ECPO）、arXiv:2607.04242（ProGPO）、arXiv:2511.12159v1（CriticSearch）、arXiv:2606.12634（SGCD）、arXiv:2606.16215（PACT）、arXiv:2601.18779（POPE）、arXiv:2607.19313（OC-GRPO）、arXiv:2605.12004（ActGuide-RL）、arXiv:2605.15155v1、arXiv:2606.17250。

hint curriculum：arXiv:2604.07747（DAHS+BHA）、arXiv:2512.13095（ADHint）、arXiv:2509.06923v1（SEELE）。

奖励设计与工程现实：arXiv:2507.17746（RaR）、arXiv:2505.14069v1、arXiv:2511.19399（DR Tulu/RLER）、arXiv:2607.24280（MAPD）、arXiv:2603.25562、arXiv:2604.08880、arXiv:2602.03468（IntentRL）、arXiv:2604.03098、support.claude.com/en/articles/12326764、anthropic.com/news/detecting-and-preventing-distillation-attacks。

来源成熟度总体提示：绝大多数为未经同行评审的 arXiv preprint，多为单实验室、单 benchmark、无独立复现。已知有正式记录的包括 SAGE-Agent（ACL Findings 2026）、DeepVerifier（Findings of ACL 2026）、DR Tulu（ICML 2026 poster）、To Distill or Decide（NeurIPS 2025）、Provable POMDP with Privileged Information（NeurIPS 2024）、Sharpening Mechanism（ICLR 2025）、GiGPO（多版本修订至 2025-10）。


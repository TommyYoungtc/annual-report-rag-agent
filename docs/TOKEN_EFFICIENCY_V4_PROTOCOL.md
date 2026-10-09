# Token-Aware / Cost-Aware RAG V4 实验协议

## 1. 研究问题

在答案准确率、无答案拒答率、Gold Evidence Recall、引用 Chunk 精度和引用页码集合准确率均不下降的条件下，比较模型路由、Prompt 压缩、检索上下文裁剪以及三者组合能否降低生成输入 Token、归一化成本和单个正确答案成本。

## 2. 与 V3 的隔离

- V3 冻结测试、配置锁和结果均为只读制品。
- V4 第一阶段只使用旧 Dev 与既有 Adaptive System C 轨迹。
- 不使用 V3 冻结测试调参，也不把 V3 再次运行包装成新结果。
- V4 候选锁定后才建立新的独立 Holdout；最终最多成功运行一次。

## 3. 两级证据标准

### 3.1 Tokenizer 精确回放

使用配置中指定模型的原生 tokenizer，并在应用 chat template 后计数。该阶段可以验证：

- Prompt 和上下文压缩的实际 Token 变化；
- 上下文裁剪后规则答案器的答案、拒答与引用是否退化；
- 不同模型路由策略下的归一化 Token 成本。

它不能证明 Prompt 压缩或模型路由后 LLM 质量不下降，因为回放答案仍由确定性答案器生成。报告必须带有 `llm_generation_validated=false`。

### 3.2 Provider 实测

接入固定的 OpenAI-compatible 生成接口后，以 API 返回的 `usage` 为计费事实，包括 input、output 和 cached input Token。美元费用只能在模型与价格表均明确锁定时计算；否则只报告 Token 和归一化成本，不能把估算值写成实际账单。

## 4. 对比版本

| 版本 | 模型路由 | Prompt | 上下文 |
|---|---|---|---|
| Baseline | 全部 large | 完整 | Top-10 完整 Chunk |
| Routing | 自适应 small/large | 完整 | Top-10 完整 Chunk |
| Prompt | 全部 large | 压缩 | Top-10 完整 Chunk |
| Context | 全部 large | 完整 | Top-10 证据窗口，最多 4,096 Token |
| Combined | 自适应 small/large | 压缩 | Top-10 证据窗口，最多 4,096 Token |

先做单变量消融，再评估组合方案，不展开 3×3×3 全网格。

## 5. Token 账本

每次模型调用记录：stage、model、usage source、input/output/cached input Token、静态 Prompt Token、问题 Token、裁剪前后证据 Token、截断前后 Token、调用次数与费用。`provider_usage`、`tokenizer_count`、`estimate` 三种来源不得混用。

同时记录：检索轮数、延迟、本地 GPU 峰值、Embedding 查询 Token 和 Reranker pair Token。后两项属于本地计算代理，不能计入 API 账单。

## 6. 质量门槛

相对 Baseline，以下指标观察值均不得下降：

- 端到端精确答案准确率；
- 困难无答案拒答准确率；
- 全部 Gold Evidence Recall；
- 引用 Chunk 精度；
- 引用页码集合准确率。

实际生成实验固定 temperature=0、Prompt SHA-256、模型版本与价格表。若供应商无法保证完全确定性，则每个版本至少运行三个种子，并报告均值、逐题差异和 bootstrap 置信区间。

## 7. 成本指标

- 平均 input/output/total Token；
- 上下文压缩率；
- small/large 路由比例；
- 每个正确答案 Token；
- 每个正确答案归一化成本；
- 配置价格后再报告每个正确答案 USD；
- 平均与 P95 延迟。

归一化成本定义为：

`model_multiplier × (input_tokens × input_weight + output_tokens × output_weight)`

它只用于无供应商价格时的相对比较。

## 8. 封存条件

开发门槛全部通过后，固化代码、模型 ID、Prompt、tokenizer、价格表、语料、数据集、上下文策略和路由阈值的 SHA-256。锁定后不得依据 Holdout 结果修改参数。

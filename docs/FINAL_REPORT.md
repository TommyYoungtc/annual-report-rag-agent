# 年报研究 Agent 最终报告

## 最终结论

项目已按 `annual-report-agent-final-v1` 配置封顶。最终系统使用微调后的
`bge-small-zh-v1.5` 进行 Dense Retrieval，使用 `Qwen3-Reranker-0.6B`
重排，并通过公司/年份路由、查询约束和确定性答案抽取器生成带页码引用的答案。

配置在冻结测试前锁定，模型权重、向量缓存、语料、评测脚本和核心回答代码均记录
SHA-256。冻结测试只成功运行一次，未因结果修改测试集、配置或重新调参。

## 数据与模型

- 真实语料：3 家上市公司、2024/2025 年共 6 份年报，2,775 个 Chunk。
- 开发集：60 题；困难无答案开发集：18 题。
- 微调数据：48 个训练问题、1,152 条难负例三元组。
- 微调验证集：12 题，与训练问题隔离。
- 冻结测试集：30 题。
- Embedding：微调 `BAAI/bge-small-zh-v1.5`，2 epochs。
- Reranker：`Qwen3-Reranker-0.6B`。
- 硬件：RTX 4060 Laptop 8GB。

## 最终指标

| 阶段 | 指标 | 结果 |
|---|---|---:|
| 微调留出验证 | Dense Recall@1 | 91.7% |
| 微调留出验证 | Dense Recall@10 | 100% |
| Dev 主链路 | 精确答案准确率 | 100% |
| Dev 主链路 | 全部 Gold 证据 Recall@10 | 100% |
| Dev 主链路 | 引用 Chunk 精度 / 页码准确率 | 100% / 100% |
| 困难无答案 Dev | 约束感知拒答准确率 | 100%（18/18） |
| 冻结 Test | 全部 Gold 证据 Recall@10 | 96.7%（29/30） |
| 冻结 Test | 回答后的引用 Chunk 精度 | 100% |
| 冻结 Test | 端到端精确答案准确率 | 20.0%（6/30） |
| 冻结 Test | Oracle Gold 证据答案准确率 | 20.0% |

训练耗时 24.5 秒，训练峰值分配显存 1.59 GiB。最终 Agent 回归时，
Embedding 峰值约 55 MiB，Reranker 峰值约 1.75 GiB。

## 冻结测试 Bad Case

冻结测试中的 6 道跨年份营业收入/研发投入题全部回答正确。其余 24 题被受控回答器以
`unsupported_fact` 拒答，分别属于以下 4 个仅在冻结测试出现的字段：

- 基本每股收益 `basic_eps`：0/6；
- 加权平均净资产收益率 `roe`：0/6；
- 资产总额 `total_assets`：0/6；
- 研发人员数 `rnd_staff`：0/6。

其中 23/24 道拒答题的 Gold 证据已经进入重排 Top-10。即使直接把 Gold Chunk
交给答案器，整体准确率仍为 20%，因此主要失败来自答案字段 Schema 覆盖，而不是检索。
唯一检索 miss 是 `test-019`。

该结果说明：难负例微调显著改善了域内检索，但规则式受控答案器无法自动泛化到未注册字段。
如果开发 v2，应新建独立测试集后再扩展字段注册表或改用带 Schema 约束的生成式抽取；
不能在本次冻结测试上修复后重新报告同一套 Test。

## 工程闭环

- 68 个自动化测试通过，Ruff 全项目通过。
- FastAPI 真实请求验证了本地 GPU、答案、引用页码和越界拒答。
- 最终配置：[final_v1.yaml](../configs/final_v1.yaml)。
- 配置锁：[final_v1.lock.json](../configs/final_v1.lock.json)。
- 结果封印：[final_v1_result.json](../configs/final_v1_result.json)。
- 完整原始结果位于本地 `outputs/frozen_test_final_v1.json`，由 `.gitignore` 排除，
  其 SHA-256 记录在结果封印中。

## 可复现命令

```powershell
python scripts/build_embedding_training_data.py
python scripts/validate_embedding_training_data.py
python scripts/train_embedding_model.py
python scripts/run_agent_evaluation.py --output outputs/agent_dev_metrics_bge_final_v1.json
python scripts/run_hard_no_answer_evaluation.py
python scripts/freeze_final_configuration.py
```

冻结 Test 命令不应再次执行；一次性门禁会因结果文件已经存在而拒绝运行。

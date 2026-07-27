# 受控 Agent 路由与范围守卫

## 目标

检索前先对问题做确定性解析，避免系统对明显超出语料范围的问题强行生成答案。当前语料范围由
`data/processed/pypdf_corpus.jsonl` 自动统计，不在代码中手写：

- 公司：宁德时代、比亚迪、科大讯飞；
- 年份：2024、2025。

## 路由结果

`src/annual_report_agent/agent/router.py` 输出结构化 `QueryRoute`：

- `companies`：问题中命中的已收录公司；
- `years`：问题中的四位年份；
- `task_type`：`single_fact`、`cross_year`、`cross_company`、`calculation` 或
  `out_of_scope`；
- `requires_calculation`：是否出现同比、增长、差额等计算意图；
- `should_refuse` 和 `refusal_reason`：是否拒答及原因；
- `requested_company`：识别到的未收录公司。

范围守卫目前处理两种可确定拒答：

1. `period_not_covered`：问题包含语料范围之外的年份；
2. `company_not_covered`：问题明确指定未收录公司。

拒答文本会说明当前覆盖范围，不编造财务数字，也不把模型知识当成年报证据。

## 开发阶段评测

复现命令：

```powershell
python scripts/evaluate_scope_guard.py
```

输出：`outputs/scope_guard_metrics_v2.json`。

| 数据 | 数量 | 指标 | 结果 |
|---|---:|---|---:|
| Dev | 60 | 接受率 | 100% |
| No-answer | 15 | 拒答准确率 | 100% |
| No-answer | 15 | 拒答原因准确率 | 100% |

本实验没有运行30题冻结 Test。No-answer 集包含9道年份越界和6道公司越界题，规则与样例边界
较清晰，因此 100% 不能解释为通用无答案准确率。

## 当前限制

- 尚未判断“公司和年份在范围内，但文档没有披露该字段”的语义无答案问题；
- 未做公司别名、证券代码和口语表达归一化；
- 单一正则路由不能替代检索置信度与证据充分性判断；
- 路由只决定检索/拒答路径，不能直接生成最终答案。

下一阶段会把路由接到 Hybrid + Reranker，并增加证据阈值、数字计算和页码引用。届时应新增
一组“范围内但无证据”的困难无答案开发题，单独报告误拒答率与漏拒答率。

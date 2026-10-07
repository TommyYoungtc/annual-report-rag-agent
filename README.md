# 上市公司年报研究 Agent

这是一个以评测为中心的中文年报 RAG/Agent 项目。项目目标不是制作一个聊天界面，而是建立一条可复现、可比较、可分析失败原因的完整链路：

```text
PDF解析 → 结构化分块 → BM25/Dense混合召回 → Reranker
→ 受控工具调用 → 带页码引用的回答 → 离线评测与消融实验
```

## V2.1 真实盲测结果（当前推荐版本）

V2.1 使用 7 家上市公司 2024/2025 年共 14 份年报，合计 4,076 页、
6,831 个 Chunk。开发阶段使用 5 家公司，完成 122 题 Dev 和 30 题困难无答案
Dev；随后先锁定源码、模型、检索参数和开发结果，再选择从未进入开发集的
立讯精密、迈瑞医疗作为公司级留出集。

| 数据划分 | 指标 | 结果 |
|---|---|---:|
| 122 题 Dev | 端到端精确答案 / Gold Recall@10 | 100% / 100% |
| 30 题困难无答案 Dev | 拒答准确率 | 100% |
| 32 题冻结盲测 | 端到端精确答案 | **93.75%（30/32）** |
| 32 题冻结盲测 | 全部 Gold 证据 Recall@10 | **100%（32/32）** |
| 32 题冻结盲测 | 引用 Chunk / 页码准确率 | 93.75% / 93.75% |
| 32 题 Gold Oracle | 答案精确匹配 | 93.75%（30/32） |

冻结盲测只成功运行一次，且运行后没有修改核心代码、参数或测试集，也没有二次测试。
两道错误均发生在立讯精密 2025 年：正确年度证据已进入重排 Top-10，但答案器优先选择了
“分季度主要财务指标”页中的第一季度净利润和现金流量数字。完整时间线、哈希和泄露检查见
[V2_EVALUATION_AUDIT.md](docs/V2_EVALUATION_AUDIT.md)。

> 早期曾在海康威视、美的集团上得到一份 100% 的 V2 结果，但这两家公司已被用于
> 错误分析和规则迭代，因此该结果属于开发集结果，已明确作废，不能作为测试集成绩。

当前版本已经包含：

- 结构感知 Markdown 分块器；
- 不依赖第三方库的 BM25 基线；
- Reciprocal Rank Fusion（RRF）混合召回；
- Sentence Transformers/BGE/Qwen3 Dense Retrieval 适配器；
- 公司/年份元数据过滤与年报章节查询扩展；
- 可复用的文档向量缓存与加权 RRF；
- 顺序卸载 Embedding 后再加载 Qwen3 原生 yes/no-logit scorer 的 8GB 显存安全重排链路；
- Recall@K、MRR、nDCG@K 评测；
- 14份真实年报的数据清单与断点安全下载脚本；
- 122题开发集、32题公司级冻结盲测和30题困难无答案开发集；
- 确定性问题路由、公司/年份范围守卫与结构化拒答；
- Hybrid + Reranker证据回答、Decimal计算和Chunk/页码/原文引用；
- 18题范围内困难无答案集与查询约束—证据蕴含检查；
- FastAPI接口、本地Web演示与8GB显存安全的串行模型运行时；
- 可离线运行的合成年报样例；
- BGE-small 难负例微调、配置哈希锁与一次性冻结测试门禁；
- 76 个自动化测试和完整 Bad Case 报告。

最终结果与限制见 [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)，简历和面试表述见
[docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md)，可直接放入个人 skill 的项目描述见
[SKILL_PROJECT_DESCRIPTION_V2.md](docs/SKILL_PROJECT_DESCRIPTION_V2.md)。

## 硬件策略

本项目按 RTX 4060 8GB 设计：

- PDF 逐份解析，不并发加载；
- 微调 BGE-small 和 Qwen3-Reranker-0.6B 分时加载；
- Embedding 最大长度默认 384，batch size 默认 32；
- Reranker 每批 1～2 条；
- 不进行 7B 模型训练；
- 训练环节使用小型 Embedding 模型；
- 生成模型优先走 OpenAI 兼容 API。

## 1. 快速运行

只运行当前纯 Python 基线，不需要安装任何机器学习依赖：

```powershell
python scripts/prepare_sample_data.py
python scripts/run_baseline.py
python scripts/run_evaluation.py
python -m unittest discover -s tests -t . -v
```

检查本机环境：

```powershell
python scripts/check_environment.py
```

下载真实年报：

```powershell
python scripts/download_reports.py --dry-run
python scripts/download_reports.py
python -m pip install -e ".[documents]"
python scripts/parse_reports_pypdf.py
```

真实数据清单位于 `data/raw/manifest.jsonl`，当前覆盖：

- 宁德时代：2024、2025；
- 科大讯飞：2024、2025；
- 比亚迪：2024、2025。

PDF体积较大且不会提交Git。下载器会校验PDF文件头、最低大小并输出SHA-256。
`parse_reports_pypdf.py` 提供轻量、可复现的逐页文本基线；MinerU解析结果将在
第二组实验中与它比较，重点评估表格、标题和阅读顺序的差异。

## 2. GPU模型环境

当前机器按 RTX 4060 Laptop 8GB 配置。建议单独创建项目环境，并先根据
[PyTorch官方安装页](https://pytorch.org/get-started/locally/)安装Windows CUDA版
PyTorch，再安装项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先安装官方CUDA版PyTorch，然后：
python -m pip install -e ".[models,dev]"
python scripts/check_environment.py
```

项目脚本默认将Hugging Face和Torch缓存放在项目内的 `cache/`，避免模型文件
散落到用户目录；该目录已被Git忽略。

运行Qwen3 Dense Retrieval样例：

```powershell
python scripts/run_dense_evaluation.py --model Qwen/Qwen3-Embedding-0.6B
```

运行当前真实年报开发集实验（本机使用已下载的 ModelScope 模型）：

```powershell
python scripts/run_dense_evaluation.py `
  --model .\cache\models\Qwen3-Embedding-0.6B-modelscope `
  --device cuda --batch-size 4 --max-length 768 `
  --top-k 10 --candidate-k 30 `
  --metadata-filter --query-expansion --bm25-weight 10 `
  --embedding-cache .\cache\embeddings\pypdf_qwen3_0.6b_768.npz `
  --corpus .\data\processed\pypdf_corpus.jsonl `
  --eval .\data\eval\annual_report_eval_v1.jsonl `
  --output .\outputs\real_dense_metrics_v1_weight10.json
```

当前真实开发集包含 2,775 个 Chunk 和 24 道人工核验问题。缓存命中后的
索引阶段约 1.2 秒，RTX 4060 Laptop 峰值显存约 1.15GB。完整实验和限制见
`docs/EXPERIMENT_LOG.md`；这些开发集数字不能直接当作最终测试集成绩。

运行 Reranker 消融（需要先将模型放到下述本地目录）：

```powershell
python scripts/run_reranker_evaluation.py `
  --embedding-model .\cache\models\Qwen3-Embedding-0.6B-modelscope `
  --reranker-model .\cache\models\Qwen3-Reranker-0.6B-modelscope `
  --embedding-cache .\cache\embeddings\pypdf_qwen3_0.6b_768.npz `
  --candidate-k 30 --rerank-candidates 10 --top-k 10 `
  --reranker-batch-size 1 --reranker-max-length 1024
```

脚本会先生成全部 Hybrid 候选，然后显式卸载 Embedding 模型并清理 CUDA
缓存，最后加载 Reranker，避免两套 0.6B 模型同时占用显存。

当前开发集上，Top-10 重排保持 Recall@5/10 为 1.00，Recall@1 从 0.5556
提升到 0.5972，MRR 从 0.7569 提升到 0.7806；P95 重排延迟约 1.57 秒，
峰值分配显存约 1.74GB。该数字来自 24 题开发集，只用于配置选择。

构建并验证正式 Eval v2：

```powershell
python scripts/build_eval_v2.py
python scripts/validate_eval_v2.py
```

Eval v2 共105题：60题 Dev、30题冻结 Test、15题 No-answer。开发阶段只允许
使用 Dev 调参；Test 和 No-answer 的 SHA-256 记录在
`data/eval/eval_v2_manifest.json`，详见 `docs/EVAL_SET_V2.md`。

运行受控 Agent 的范围守卫评测（只使用 Dev 与 No-answer，不运行冻结 Test）：

```powershell
python scripts/evaluate_scope_guard.py
```

当前范围守卫在60题 Dev上接受率为100%，在15题 No-answer上拒答与原因分类准确率均为
100%。这些无答案题只覆盖公司/年份越界，不代表系统已经解决“范围内但缺少证据”的语义拒答；
完整说明见 `docs/AGENT_ROUTING.md`。

运行端到端证据回答评测：

```powershell
python scripts/run_agent_evaluation.py
```

如果只修改确定性证据选择逻辑，可复用已保存的Reranker输出：

```powershell
python scripts/replay_agent_answers.py
```

当前60题 Dev的答案精确匹配、引用Chunk精度和页码准确率均为100%，15题 No-answer全部拒答。
这些规则使用了Dev错误案例迭代，不能当成冻结测试成绩；GPU、延迟、初始错误与限制见
`docs/ANSWER_PIPELINE.md`。

构建、验证并运行范围内困难无答案实验：

```powershell
python scripts/build_hard_no_answer.py
python scripts/validate_hard_no_answer.py
python scripts/run_hard_no_answer_evaluation.py
```

18题困难集中，不加语义约束时系统会18题全部误答；查询约束—证据蕴含检查可拒绝18/18且不影响
60题Dev。单一Reranker阈值无法达到同样权衡，详见 `docs/HARD_NO_ANSWER.md`。

启动本地演示：

```powershell
python -m pip install -e ".[api]"
python scripts/run_demo.py
```

浏览器打开 `http://127.0.0.1:8000`。页面会展示路由、答案、计算过程、引用页码、证据原文、
拒答原因和请求耗时。服务保持Embedding/Reranker串行加载以适配RTX 4060 8GB，详见
`docs/LOCAL_DEMO.md`。

或者安装为可编辑包：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## 3. 样例输出

`run_baseline.py` 会输出每个问题的 Top-3 检索结果。`run_evaluation.py` 会输出：

- Recall@1
- Recall@3
- MRR
- nDCG@3

样例数据是为了验证代码而构造的虚拟年报片段，不代表任何真实上市公司的披露。

## 4. 目录结构

```text
annual-report-agent/
├── configs/                 # 模型与检索配置
├── data/
│   ├── raw/                 # 原始PDF，不提交Git
│   ├── parsed/              # pypdf/MinerU解析结果
│   ├── processed/           # Chunk与索引
│   ├── eval/                # 冻结评测集
│   └── samples/             # 可公开的合成样例
├── outputs/                 # 指标、日志和Bad Case
├── scripts/                 # 数据准备和实验入口
├── src/annual_report_agent/
│   ├── ingestion/           # 解析与分块
│   ├── retrieval/           # BM25、Dense、RRF、Reranker
│   ├── agent/               # 问题路由、范围守卫与受控工具调用
│   └── evaluation/          # 检索与答案评测
└── tests/
```

## 5. 数据规范

处理后的每个 Chunk 使用如下 JSONL 格式：

```json
{
  "chunk_id": "demo-tech-2024:0001",
  "document_id": "demo-tech-2024",
  "company": "示例科技",
  "year": 2024,
  "section": "研发投入",
  "page": 12,
  "text": "……"
}
```

评测问题格式：

```json
{
  "query_id": "q001",
  "query": "示例科技2024年的研发投入是多少？",
  "relevant_chunk_ids": ["demo-tech-2024:0001"],
  "question_type": "single_fact"
}
```

## 6. 实验纪律

1. 测试集一旦冻结，不允许因结果不好而修改。
2. LLM 生成的问题必须经过人工抽查。
3. 检索和回答指标分开报告。
4. 每次改动记录配置、随机种子和结果。
5. 不只展示最好结果，同时保留失败案例。
6. 简历中只填写真实测得的数字。

## 7. 当前里程碑

查看 [PROJECT_PLAN.md](PROJECT_PLAN.md)、[docs/WEEK1_CHECKLIST.md](docs/WEEK1_CHECKLIST.md)、
[docs/EVAL_SET_V1.md](docs/EVAL_SET_V1.md) 和
[docs/EVAL_SET_V2.md](docs/EVAL_SET_V2.md)、
[docs/AGENT_ROUTING.md](docs/AGENT_ROUTING.md) 和
[docs/ANSWER_PIPELINE.md](docs/ANSWER_PIPELINE.md)、
[docs/HARD_NO_ANSWER.md](docs/HARD_NO_ANSWER.md)、
[docs/LOCAL_DEMO.md](docs/LOCAL_DEMO.md)、
[docs/EMBEDDING_FINETUNING.md](docs/EMBEDDING_FINETUNING.md)、
[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)、
[docs/ADAPTIVE_RETRIEVAL_EXPERIMENT.md](docs/ADAPTIVE_RETRIEVAL_EXPERIMENT.md)、
[docs/ADAPTIVE_RETRIEVAL_DEV_REPORT.md](docs/ADAPTIVE_RETRIEVAL_DEV_REPORT.md)、
[docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md) 和
[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)。

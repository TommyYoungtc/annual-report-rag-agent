# BGE-small 年报检索难负例微调

## 实验结论

在 12 题微调留出验证集上，纯稠密检索 Recall@1 从 20.8% 提升到 91.7%，Recall@10 从 54.2% 提升到 100.0%。

| 检索器 | 指标 | 微调前 | 微调后 | 绝对变化 |
|---|---:|---:|---:|---:|
| dense | recall@1 | 20.8% | 91.7% | 70.8% |
| dense | recall@3 | 45.8% | 100.0% | 54.2% |
| dense | recall@5 | 45.8% | 100.0% | 54.2% |
| dense | recall@10 | 54.2% | 100.0% | 45.8% |
| dense | mrr | 37.5% | 100.0% | 62.5% |
| dense | ndcg@10 | 39.4% | 100.0% | 60.6% |
| hybrid | recall@1 | 54.2% | 75.0% | 20.8% |
| hybrid | recall@3 | 87.5% | 100.0% | 12.5% |
| hybrid | recall@5 | 91.7% | 100.0% | 8.3% |
| hybrid | recall@10 | 100.0% | 100.0% | 0.0% |
| hybrid | mrr | 72.0% | 91.7% | 19.6% |
| hybrid | ndcg@10 | 78.6% | 93.8% | 15.3% |

## 数据与训练设置

- 原始开发问题：60；固定按源顺序每第 5 题留出，训练 48、验证 12。
- 难负例：每个训练问题 24 条，共 1,152 条三元组。
- 安全校验：训练/验证泄漏 0，gold/相邻证据误标负例 0，答案证据误标负例 0。
- 基座：BAAI/bge-small-zh-v1.5；损失：MultipleNegativesRankingLoss。
- 训练：2 epochs，batch=16，max_length=384，lr=2e-05。
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU；训练 24.5s，峰值分配显存 1591.2 MiB，峰值预留 1710.0 MiB。

## Bad Case 与边界

- 微调后 12 题的所有标注证据均进入 Dense Top-10，没有剩余 Top-10 miss。
- Recall@1 未达到 100% 的原因是跨年份问题各有两个 gold chunk；Top-1 最多覆盖其中一个，不代表首个相关证据没有排在第一。
- 这 12 题未参与微调，但与训练题来自相同的 6 份年报和同一题型族，因此结果证明域内适配有效，不能替代最终冻结测试集。
- 冻结测试集仍未运行，需在模型与检索配置最终锁定后只执行一次。

## 复现

```powershell
python -m pip install -e ".[models,training]"
python scripts/build_embedding_training_data.py
python scripts/validate_embedding_training_data.py
python scripts/download_embedding_model.py
python scripts/train_embedding_model.py --epochs 2 --batch-size 16 --max-length 384
```

评测时对微调前后模型使用完全相同的语料、查询前缀、metadata filter、
BM25 query expansion 与融合参数；两个完整 JSON 结果保存在 `outputs/`。

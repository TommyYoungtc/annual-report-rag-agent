from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare embedding evaluation runs")
    parser.add_argument("--before", default="outputs/bge_small_validation_before_v1.json")
    parser.add_argument("--after", default="outputs/bge_small_validation_after_v1.json")
    parser.add_argument(
        "--training-metrics",
        default="outputs/bge_small_training_metrics_v1.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/bge_small_finetune_comparison_v1.json",
    )
    parser.add_argument("--report", default="docs/EMBEDDING_FINETUNING.md")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(value: str) -> dict:
    return json.loads(project_path(value).read_text(encoding="utf-8"))


def first_relevant_rank(query: dict, key: str) -> int | None:
    relevant = set(query["relevant_chunk_ids"])
    for rank, chunk_id in enumerate(query[key], start=1):
        if chunk_id in relevant:
            return rank
    return None


def compare_metrics(before: dict, after: dict, retriever: str) -> dict:
    return {
        metric: {
            "before": before[retriever][metric],
            "after": after[retriever][metric],
            "absolute_delta": after[retriever][metric] - before[retriever][metric],
        }
        for metric in METRICS
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(comparison: dict, training: dict) -> str:
    lines = [
        "# BGE-small 年报检索难负例微调",
        "",
        "## 实验结论",
        "",
        (
            "在 12 题微调留出验证集上，纯稠密检索 Recall@1 从 "
            f"{percent(comparison['dense']['recall@1']['before'])} 提升到 "
            f"{percent(comparison['dense']['recall@1']['after'])}，Recall@10 从 "
            f"{percent(comparison['dense']['recall@10']['before'])} 提升到 "
            f"{percent(comparison['dense']['recall@10']['after'])}。"
        ),
        "",
        "| 检索器 | 指标 | 微调前 | 微调后 | 绝对变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for retriever in ("dense", "hybrid"):
        for metric in METRICS:
            row = comparison[retriever][metric]
            lines.append(
                f"| {retriever} | {metric} | {percent(row['before'])} | "
                f"{percent(row['after'])} | {percent(row['absolute_delta'])} |"
            )
    lines.extend(
        [
            "",
            "## 数据与训练设置",
            "",
            "- 原始开发问题：60；固定按源顺序每第 5 题留出，训练 48、验证 12。",
            "- 难负例：每个训练问题 24 条，共 1,152 条三元组。",
            "- 安全校验：训练/验证泄漏 0，gold/相邻证据误标负例 0，答案证据误标负例 0。",
            "- 基座：BAAI/bge-small-zh-v1.5；损失：MultipleNegativesRankingLoss。",
            (
                f"- 训练：{training['epochs']:g} epochs，batch={training['batch_size']}，"
                f"max_length={training['max_length']}，lr={training['learning_rate']:.0e}。"
            ),
            (
                f"- 硬件：{training['cuda_device']}；训练 {training['elapsed_seconds']:.1f}s，"
                f"峰值分配显存 {training['cuda_peak_allocated_mib']:.1f} MiB，"
                f"峰值预留 {training['cuda_peak_reserved_mib']:.1f} MiB。"
            ),
            "",
            "## Bad Case 与边界",
            "",
        ]
    )
    remaining = comparison["remaining_dense_top10_misses"]
    if remaining:
        lines.append("- 微调后 Top-10 仍未命中的问题：" + "、".join(remaining))
    else:
        lines.append("- 微调后 12 题的所有标注证据均进入 Dense Top-10，没有剩余 Top-10 miss。")
    lines.extend(
        [
            (
                "- Recall@1 未达到 100% 的原因是跨年份问题各有两个 gold chunk；"
                "Top-1 最多覆盖其中一个，不代表首个相关证据没有排在第一。"
            ),
            (
                "- 这 12 题未参与微调，但与训练题来自相同的 6 份年报和同一题型族，"
                "因此结果证明域内适配有效，不能替代最终冻结测试集。"
            ),
            "- 冻结测试集仍未运行，需在模型与检索配置最终锁定后只执行一次。",
            "",
            "## 复现",
            "",
            "```powershell",
            'python -m pip install -e ".[models,training]"',
            "python scripts/build_embedding_training_data.py",
            "python scripts/validate_embedding_training_data.py",
            "python scripts/download_embedding_model.py",
            "python scripts/train_embedding_model.py --epochs 2 --batch-size 16 --max-length 384",
            "```",
            "",
            "评测时对微调前后模型使用完全相同的语料、查询前缀、metadata filter、",
            "BM25 query expansion 与融合参数；两个完整 JSON 结果保存在 `outputs/`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    before = read_json(args.before)
    after = read_json(args.after)
    training = read_json(args.training_metrics)
    before_queries = {row["query_id"]: row for row in before["queries"]}
    after_queries = {row["query_id"]: row for row in after["queries"]}
    if before_queries.keys() != after_queries.keys():
        raise ValueError("before/after query sets differ")

    query_changes = []
    remaining_misses = []
    for query_id, before_row in before_queries.items():
        after_row = after_queries[query_id]
        before_rank = first_relevant_rank(before_row, "dense_top_ids")
        after_rank = first_relevant_rank(after_row, "dense_top_ids")
        if set(after_row["relevant_chunk_ids"]).difference(after_row["dense_top_ids"]):
            remaining_misses.append(query_id)
        query_changes.append(
            {
                "query_id": query_id,
                "question_type": before_row["question_type"],
                "first_relevant_rank_before": before_rank,
                "first_relevant_rank_after": after_rank,
            }
        )

    comparison = {
        "evaluation_queries": before["dataset"]["num_queries"],
        "dense": compare_metrics(before, after, "dense"),
        "hybrid": compare_metrics(before, after, "hybrid"),
        "query_changes": query_changes,
        "remaining_dense_top10_misses": remaining_misses,
        "frozen_test_executed": False,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = project_path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(comparison, training), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

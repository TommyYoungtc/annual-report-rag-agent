# 第一周任务清单

## Day 1：环境与数据选择

- [x] 运行 `nvidia-smi` 并保存 GPU、驱动和显存信息；
- [x] 记录 CPU、内存和可用磁盘；
- [x] 确定 3 家公司，每家公司选择连续 2 年；
- [x] 创建 `data/raw/manifest.jsonl`；
- [x] 记录每份年报的来源 URL、公司和年份。

## Day 2～3：MinerU 解析

- [x] 安装符合本机 CUDA 的 PyTorch；
- [ ] 单独环境安装 MinerU；
- [ ] 使用 pipeline 后端逐份解析；
- [x] 使用 pypdf 完成逐页基线解析并输出 Markdown/JSON；
- [x] 将 Markdown/JSON 放入 `data/parsed/pypdf/`；
- [x] 自动检查 1,609 页空页数并人工核对关键证据页；
- [ ] 记录 OCR、页码、表格和页眉页脚问题。

## Day 4：结构化分块

- [x] 将 pypdf Markdown 转成 Chunk JSONL；
- [x] 确认公司、年份、章节、页码没有丢失；
- [ ] 对表格进行整体保护；
- [x] 输出每份文档页数、字符数和 Chunk 数。

## Day 5：BM25 基线

- [x] 对处理后的 Chunk 建立 BM25 索引；
- [x] 准备 24 道开发问题；
- [x] 输出 Top-10 结果；
- [x] 标记每题相关 Chunk并核对答案数字。

## Day 6：指标与错误分析

- [x] 计算 Recall@1/3/5/10、MRR、nDCG@10；
- [x] 分析 6 个首轮典型检索错误；
- [x] 将主要错误定位为高频字段召回和跨页表格问题；
- [x] 用元数据过滤、章节查询扩展和加权 RRF 完成首轮修复。

## Day 7：周验收

- [x] 新环境可以按 README 跑通样例；
- [x] 完成第一周实验结果与复现命令；
- [x] 确认第二周优先项：评测集扩充、Reranker、语义/无答案问题；
- [x] 冻结第一版数据规范。

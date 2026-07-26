from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.ingestion import MarkdownDocument, chunk_markdown
from annual_report_agent.io_utils import write_chunks


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_pdf(path: Path) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            'pypdf is required. Install with: python -m pip install -e ".[documents]"'
        ) from error

    started = time.perf_counter()
    reader = PdfReader(path)
    blocks: list[str] = []
    empty_pages = 0
    extracted_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages += 1
        extracted_chars += len(text)
        blocks.append(f"<!-- page: {page_number} -->\n{text}")
    return "\n\n".join(blocks), {
        "pages": len(reader.pages),
        "empty_pages": empty_pages,
        "extracted_chars": extracted_chars,
        "seconds": round(time.perf_counter() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse downloaded annual reports with pypdf")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "raw" / "manifest.jsonl",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=ROOT / "data" / "parsed" / "pypdf",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "processed" / "pypdf_corpus.jsonl",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.only:
        selected = set(args.only)
        rows = [row for row in rows if row["document_id"] in selected]

    args.parsed_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    reports = []
    for row in rows:
        pdf_path = args.raw_dir / row["filename"]
        if not pdf_path.exists():
            print(f"SKIP missing PDF: {pdf_path}", file=sys.stderr)
            continue
        markdown, stats = extract_pdf(pdf_path)
        markdown_path = args.parsed_dir / f"{row['document_id']}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        chunks = chunk_markdown(
            MarkdownDocument(
                document_id=row["document_id"],
                company=row["company"],
                year=int(row["year"]),
                markdown=markdown,
            ),
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
        )
        all_chunks.extend(chunks)
        reports.append(
            {
                "document_id": row["document_id"],
                "pdf": str(pdf_path),
                "markdown": str(markdown_path),
                "chunks": len(chunks),
                **stats,
            }
        )
        print(json.dumps(reports[-1], ensure_ascii=False))

    if not all_chunks:
        raise SystemExit("No PDFs were parsed. Download reports first.")
    write_chunks(args.corpus, all_chunks)
    report_path = args.parsed_dir / "extraction_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks to {args.corpus}")
    print(f"Wrote extraction report to {report_path}")


if __name__ == "__main__":
    main()

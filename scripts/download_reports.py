from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "manifest.jsonl"
MIN_PDF_BYTES = 10_000


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download(row: dict, output_dir: Path, *, dry_run: bool = False) -> dict:
    target = output_dir / row["filename"]
    if is_valid_pdf(target):
        return {
            "document_id": row["document_id"],
            "status": "already_exists",
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    if dry_run:
        return {
            "document_id": row["document_id"],
            "status": "dry_run",
            "path": str(target),
            "url": row["url"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        row["url"],
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        },
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if not is_valid_pdf(temporary):
            raise RuntimeError(f"Downloaded file is not a valid PDF: {temporary}")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "document_id": row["document_id"],
        "status": "downloaded",
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download annual reports from manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--only", action="append", default=[], help="document_id to download")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.only:
        selected = set(args.only)
        rows = [row for row in rows if row["document_id"] in selected]
        missing = selected - {row["document_id"] for row in rows}
        if missing:
            raise SystemExit(f"Unknown document_id(s): {', '.join(sorted(missing))}")
    if args.max_files is not None:
        rows = rows[: max(args.max_files, 0)]

    failures = 0
    for row in rows:
        try:
            result = download(row, args.output_dir, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False))
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "document_id": row["document_id"],
                        "status": "failed",
                        "error": str(error),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
    if failures:
        raise SystemExit(f"{failures} download(s) failed")


if __name__ == "__main__":
    main()

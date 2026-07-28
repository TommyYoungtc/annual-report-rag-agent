from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local annual-report Agent demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["ANNUAL_REPORT_PROJECT_ROOT"] = str(ROOT)
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            'API dependencies are missing. Install with: python -m pip install -e ".[api]"'
        ) from error
    uvicorn.run(
        "annual_report_agent.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

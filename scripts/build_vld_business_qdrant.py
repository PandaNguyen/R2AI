"""Backward-compatible wrapper for the VLD Qdrant artifact builder."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from r2ai.data_ingest.vld.business_qdrant import main


if __name__ == "__main__":
    main()

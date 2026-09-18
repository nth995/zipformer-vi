from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_scripts.fleurs import create_label as create_fleurs_label
from dataset_scripts.vivos import create_label as create_vivos_label
from dataset_scripts.vlsp import create_label as create_vlsp_label


def main() -> None:
    """Tạo label cho bộ dữ liệu được chọn."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["VIVOS", "VLSP", "FLEURS"],
        help="Tên bộ dữ liệu cần tạo label.",
    )
    args = parser.parse_args()

    if args.dataset == "VIVOS":
        create_vivos_label(
            raw_dir=PROJECT_ROOT / "data" / "raw" / "VIVOS",
            processed_dir=PROJECT_ROOT / "data" / "processed" / "VIVOS",
        )

    elif args.dataset == "VLSP":
        create_vlsp_label(
            raw_dir=PROJECT_ROOT / "data" / "raw" / "VLSP",
            processed_dir=PROJECT_ROOT / "data" / "processed" / "VLSP",
        )

    elif args.dataset == "FLEURS":
        create_fleurs_label(
            raw_dir=PROJECT_ROOT / "data" / "raw" / "FLEURS",
            processed_dir=PROJECT_ROOT / "data" / "processed" / "FLEURS",
        )


if __name__ == "__main__":
    main()

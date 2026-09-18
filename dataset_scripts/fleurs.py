from __future__ import annotations

import csv
import os
from pathlib import Path


SPLITS = ["train", "dev", "test"]


def find_fleurs_root(raw_dir: Path) -> Path:
    """Tìm thư mục gốc của FLEURS Vietnamese sau khi giải nén."""
    raw_dir = Path(raw_dir)

    candidates = [
        raw_dir / "FLEURS_vi_vn",
        raw_dir,
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        has_tsv = any(
            (candidate / f"{split}.tsv").exists()
            for split in SPLITS
        )

        has_audio = (candidate / "audio").exists()

        if has_tsv and has_audio:
            return candidate

    raise FileNotFoundError(
        "Không tìm thấy cấu trúc FLEURS vi_vn. "
        "Cần có FLEURS_vi_vn/train.tsv và thư mục FLEURS_vi_vn/audio/."
    )


def read_split(fleurs_root: Path, split: str) -> list[tuple[Path, str]]:
    """Đọc transcript và audio đang tồn tại trong một split."""
    tsv_path = fleurs_root / f"{split}.tsv"
    audio_dir = fleurs_root / "audio" / split

    if not tsv_path.exists():
        return []

    if not audio_dir.exists():
        return []

    samples: list[tuple[Path, str]] = []
    missing_audio = 0
    empty_label = 0

    with tsv_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line:
                continue

            columns = line.split("\t")

            if len(columns) != 7:
                raise RuntimeError(
                    f"{tsv_path}, dòng {line_number}: "
                    f"có {len(columns)} cột thay vì 7."
                )

            file_name = columns[1].strip()
            label = columns[3].strip()
            audio_path = audio_dir / file_name

            if not audio_path.exists():
                missing_audio += 1
                continue

            if not label:
                empty_label += 1
                continue

            samples.append((audio_path, label))

    if missing_audio > 0:
        print(
            f"{split}: bỏ qua {missing_audio} dòng không có audio tương ứng."
        )

    if empty_label > 0:
        print(
            f"{split}: bỏ qua {empty_label} dòng có transcript rỗng."
        )

    return samples


def read_samples(fleurs_root: Path) -> list[tuple[Path, str]]:
    """Đọc toàn bộ mẫu từ train, dev và test."""
    samples: list[tuple[Path, str]] = []

    for split in SPLITS:
        split_samples = read_split(
            fleurs_root,
            split,
        )

        samples.extend(split_samples)

        print(
            f"{split}: {len(split_samples)} audio có label."
        )

    return samples


def create_label(raw_dir: Path, processed_dir: Path) -> Path:
    """Tạo label.csv hai cột cho FLEURS Vietnamese."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fleurs_root = find_fleurs_root(raw_dir)
    samples = read_samples(fleurs_root)

    if not samples:
        raise RuntimeError(
            "Không tìm thấy audio FLEURS nào khớp với các file TSV."
        )

    output_path = processed_dir / "label.csv"

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["audio_path", "label"])

        for audio_path, label in samples:
            relative_path = os.path.relpath(
                audio_path,
                start=processed_dir,
            )

            # Dùng dấu / để CSV dùng được trên Windows và Linux.
            relative_path = Path(relative_path).as_posix()

            writer.writerow([
                relative_path,
                label,
            ])

    print(f"Đã tạo: {output_path}")
    print(f"Tổng số audio có label: {len(samples)}")

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    create_label(
        raw_dir=project_root / "data" / "raw" / "FLEURS",
        processed_dir=project_root / "data" / "processed" / "FLEURS",
    )

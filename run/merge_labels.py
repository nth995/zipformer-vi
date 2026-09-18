from __future__ import annotations

import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "train.csv"


def find_label_files(processed_dir: Path) -> list[Path]:
    """Tìm label.csv của tất cả dataset trong data/processed."""
    label_files = []

    if not processed_dir.exists():
        return label_files

    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        label_path = dataset_dir / "label.csv"

        if label_path.exists():
            label_files.append(label_path)

    return label_files


def read_label_file(label_path: Path) -> list[tuple[Path, str]]:
    """Đọc một label.csv và đổi audio_path thành đường dẫn tuyệt đối."""
    samples = []

    with label_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["audio_path", "label"]:
            raise RuntimeError(
                f"Sai cấu trúc file: {label_path}\n"
                "Cần đúng 2 cột: audio_path,label"
            )

        for line_number, row in enumerate(reader, start=2):
            audio_path = row["audio_path"].strip()
            label = row["label"].strip()

            if not audio_path or not label:
                raise RuntimeError(
                    f"{label_path}, dòng {line_number}: "
                    "audio_path hoặc label bị rỗng."
                )

            absolute_audio_path = (
                label_path.parent / audio_path
            ).resolve()

            if not absolute_audio_path.exists():
                raise FileNotFoundError(
                    f"{label_path}, dòng {line_number}: "
                    f"không tìm thấy audio:\n{absolute_audio_path}"
                )

            samples.append(
                (
                    absolute_audio_path,
                    label,
                )
            )

    return samples


def make_relative_path(
    audio_path: Path,
    output_dir: Path,
) -> str:
    """Đổi đường dẫn audio về tương đối so với file label tổng."""
    relative_path = os.path.relpath(
        audio_path,
        start=output_dir,
    )

    return Path(relative_path).as_posix()


def write_merged_label(
    output_path: Path,
    samples: list[tuple[Path, str]],
) -> None:
    """Ghi toàn bộ sample vào file label tổng."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "audio_path",
                "label",
            ]
        )

        for audio_path, label in samples:
            relative_path = make_relative_path(
                audio_path,
                output_path.parent,
            )

            writer.writerow(
                [
                    relative_path,
                    label,
                ]
            )


def merge_labels(
    processed_dir: Path,
    output_path: Path,
) -> Path:
    """Gộp label.csv của các dataset thành một file label.csv tổng."""
    label_files = find_label_files(
        processed_dir
    )

    if not label_files:
        raise RuntimeError(
            "Không tìm thấy file data/processed/<DATASET>/label.csv nào."
        )

    all_samples = []

    print("Các dataset được gộp:")

    for label_path in label_files:
        samples = read_label_file(
            label_path
        )

        dataset_name = label_path.parent.name

        print(
            f"- {dataset_name}: "
            f"{len(samples)} mẫu"
        )

        all_samples.extend(
            samples
        )

    if not all_samples:
        raise RuntimeError(
            "Không có sample nào để gộp."
        )

    write_merged_label(
        output_path,
        all_samples,
    )

    print()
    print(f"Đã tạo: {output_path}")
    print(
        f"Tổng số sample: "
        f"{len(all_samples)}"
    )

    return output_path


def main() -> None:
    """Chạy gộp toàn bộ label trong project."""
    merge_labels(
        processed_dir=PROCESSED_DIR,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

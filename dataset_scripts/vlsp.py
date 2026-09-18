from __future__ import annotations

import csv
import os
from pathlib import Path


def find_vlsp_root(raw_dir: Path) -> Path:
    """Tìm thư mục chứa các cặp WAV/TXT của VLSP2020."""
    raw_dir = Path(raw_dir)

    candidates = [
        raw_dir / "vlsp2020_train_set_02",
        raw_dir,
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        if next(candidate.rglob("*.wav"), None) is not None:
            return candidate

    raise FileNotFoundError(
        "Không tìm thấy dữ liệu VLSP2020. "
        "Cần có thư mục vlsp2020_train_set_02 chứa các file .wav và .txt."
    )


def read_transcript(txt_path: Path) -> str:
    """Đọc transcript từ file TXT mà không thay đổi nội dung câu."""
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1258",
    ]

    for encoding in encodings:
        try:
            return txt_path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Không đọc được transcript với các encoding hỗ trợ: {txt_path}"
    )


def read_samples(vlsp_root: Path) -> list[tuple[Path, str]]:
    """Đọc các cặp WAV/TXT đang tồn tại trong VLSP2020."""
    samples: list[tuple[Path, str]] = []
    missing_txt = 0
    empty_label = 0

    for audio_path in sorted(vlsp_root.rglob("*.wav")):
        txt_path = audio_path.with_suffix(".txt")

        if not txt_path.exists():
            missing_txt += 1
            continue

        label = read_transcript(txt_path)

        if not label:
            empty_label += 1
            continue

        samples.append((audio_path, label))

    if missing_txt > 0:
        print(f"Bỏ qua {missing_txt} audio không có file TXT tương ứng.")

    if empty_label > 0:
        print(f"Bỏ qua {empty_label} audio có transcript rỗng.")

    return samples


def create_label(raw_dir: Path, processed_dir: Path) -> Path:
    """Tạo label.csv hai cột cho VLSP2020."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    vlsp_root = find_vlsp_root(raw_dir)
    samples = read_samples(vlsp_root)

    if not samples:
        raise RuntimeError(
            "Không tìm thấy cặp WAV/TXT hợp lệ nào trong VLSP2020."
        )

    output_path = processed_dir / "label.csv"

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "label"])

        for audio_path, label in samples:
            relative_path = os.path.relpath(
                audio_path,
                start=processed_dir,
            )

            # Dùng dấu / để CSV dùng được trên Windows và Linux.
            relative_path = Path(relative_path).as_posix()

            writer.writerow([relative_path, label])

    print(f"Đã tạo: {output_path}")
    print(f"Tổng số audio có label: {len(samples)}")

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    create_label(
        raw_dir=project_root / "data" / "raw" / "VLSP",
        processed_dir=project_root / "data" / "processed" / "VLSP",
    )

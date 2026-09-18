from __future__ import annotations

import csv
import os
from pathlib import Path


def find_vivos_root(raw_dir: Path) -> Path:
    """Tìm thư mục gốc thật sự của VIVOS sau khi giải nén."""
    raw_dir = Path(raw_dir)

    candidates = [
        raw_dir / "vivos",
        raw_dir,
    ]

    for candidate in candidates:
        if (candidate / "train" / "prompts.txt").exists() or (
            candidate / "test" / "prompts.txt"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "Không tìm thấy cấu trúc VIVOS. "
        "Cần có train/prompts.txt hoặc test/prompts.txt."
    )


def read_split(vivos_root: Path, split: str) -> list[tuple[Path, str]]:
    """Đọc transcript và lấy các audio đang tồn tại trong một split."""
    prompt_path = vivos_root / split / "prompts.txt"

    if not prompt_path.exists():
        return []

    samples: list[tuple[Path, str]] = []

    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            audio_id, label = parts
            speaker_id = audio_id.split("_")[0]

            audio_path = (
                vivos_root
                / split
                / "waves"
                / speaker_id
                / f"{audio_id}.wav"
            )

            # Bản 100 mẫu chỉ có một phần audio nên chỉ lấy file đang tồn tại.
            if audio_path.exists():
                samples.append((audio_path, label.strip()))

    return samples


def create_label(raw_dir: Path, processed_dir: Path) -> Path:
    """Tạo label.csv hai cột cho VIVOS."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    vivos_root = find_vivos_root(raw_dir)

    samples = []
    samples.extend(read_split(vivos_root, "train"))
    samples.extend(read_split(vivos_root, "test"))

    if not samples:
        raise RuntimeError(
            "Không tìm thấy audio VIVOS nào khớp với prompts.txt."
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

            # Dùng dấu / để file CSV dùng được trên Windows và Linux.
            relative_path = Path(relative_path).as_posix()

            writer.writerow([relative_path, label])

    print(f"Đã tạo: {output_path}")
    print(f"Tổng số audio có label: {len(samples)}")

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    create_label(
        raw_dir=project_root / "data" / "raw" / "VIVOS",
        processed_dir=project_root / "data" / "processed" / "VIVOS",
    )

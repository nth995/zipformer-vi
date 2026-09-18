from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import winsound
except ImportError:
    winsound = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_FIELDS = [
    "audio_path",
    "label",
    "label_status",
    "audio_quality",
    "note",
]

LABEL_STATUS_TEXT = {
    "correct": "Đúng",
    "wrong": "Sai",
    "uncertain": "Không chắc",
}

AUDIO_QUALITY_TEXT = {
    "easy": "Dễ nghe",
    "hard": "Khó nghe",
}


class ReviewApp:
    """Giao diện nghe audio và đánh giá label."""

    def __init__(self, root: tk.Tk, label_path: Path, review_path: Path) -> None:
        self.root = root
        self.label_path = label_path
        self.review_path = review_path
        self.rows = self._load_rows()
        self.index = self._find_resume_index()

        self.label_status_var = tk.StringVar()
        self.audio_quality_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.audio_var = tk.StringVar()

        self._build_ui()
        self._load_current_row()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Left>", lambda _event: self.prev_row())
        self.root.bind("<Right>", lambda _event: self.next_row())
        self.root.bind("<space>", lambda _event: self.play_audio())

    def _load_rows(self) -> list[dict[str, str]]:
        """Đọc label.csv và ghép kết quả review cũ nếu có."""
        if not self.label_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {self.label_path}")

        with self.label_path.open("r", encoding="utf-8-sig", newline="") as f:
            label_rows = list(csv.DictReader(f))

        if not label_rows:
            raise ValueError(f"File label rỗng: {self.label_path}")

        required = {"audio_path", "label"}
        if not required.issubset(label_rows[0]):
            raise ValueError("label.csv phải có 2 cột: audio_path,label")

        old_reviews: dict[str, dict[str, str]] = {}
        if self.review_path.exists():
            with self.review_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    audio_path = (row.get("audio_path") or "").strip()
                    if audio_path:
                        old_reviews[audio_path] = row

        rows: list[dict[str, str]] = []
        for label_row in label_rows:
            audio_path = (label_row.get("audio_path") or "").strip()
            label = label_row.get("label") or ""
            old = old_reviews.get(audio_path, {})

            rows.append(
                {
                    "audio_path": audio_path,
                    "label": label,
                    "label_status": old.get("label_status", ""),
                    "audio_quality": old.get("audio_quality", ""),
                    "note": old.get("note", ""),
                }
            )

        return rows

    def _find_resume_index(self) -> int:
        """Mở lại từ mẫu đầu tiên chưa đánh giá đủ."""
        for i, row in enumerate(self.rows):
            if not row["label_status"] or not row["audio_quality"]:
                return i
        return 0

    def _build_ui(self) -> None:
        """Tạo giao diện chính."""
        self.root.title("Review label")
        self.root.geometry("900x650")
        self.root.minsize(760, 560)

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            textvariable=self.progress_var,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            main,
            textvariable=self.summary_var,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 12))

        ttk.Label(main, text="Audio:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(
            main,
            textvariable=self.audio_var,
            wraplength=840,
        ).pack(anchor="w", pady=(2, 8))

        audio_buttons = ttk.Frame(main)
        audio_buttons.pack(anchor="w", pady=(0, 14))
        ttk.Button(audio_buttons, text="▶ Phát", command=self.play_audio).pack(side="left")
        ttk.Button(audio_buttons, text="■ Dừng", command=self.stop_audio).pack(side="left", padx=(8, 0))

        ttk.Separator(main).pack(fill="x", pady=8)

        ttk.Label(main, text="Label:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.label_text = tk.Text(
            main,
            height=4,
            wrap="word",
            font=("Segoe UI", 14),
            padx=8,
            pady=8,
        )
        self.label_text.pack(fill="x", pady=(4, 14))
        self.label_text.configure(state="disabled")

        status_frame = ttk.LabelFrame(main, text="Đánh giá label", padding=10)
        status_frame.pack(fill="x", pady=(0, 10))
        for value, text in LABEL_STATUS_TEXT.items():
            ttk.Radiobutton(
                status_frame,
                text=text,
                value=value,
                variable=self.label_status_var,
                command=self._save_current_row,
            ).pack(side="left", padx=(0, 22))

        quality_frame = ttk.LabelFrame(main, text="Độ khó nghe", padding=10)
        quality_frame.pack(fill="x", pady=(0, 10))
        for value, text in AUDIO_QUALITY_TEXT.items():
            ttk.Radiobutton(
                quality_frame,
                text=text,
                value=value,
                variable=self.audio_quality_var,
                command=self._save_current_row,
            ).pack(side="left", padx=(0, 22))

        ttk.Label(main, text="Ghi chú:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.note_text = tk.Text(main, height=5, wrap="word")
        self.note_text.pack(fill="both", expand=True, pady=(4, 12))
        self.note_text.bind("<FocusOut>", lambda _event: self._save_current_row())

        nav = ttk.Frame(main)
        nav.pack(fill="x")
        ttk.Button(nav, text="← Trước", command=self.prev_row).pack(side="left")
        ttk.Button(nav, text="Lưu", command=self._save_current_row).pack(side="left", padx=8)
        ttk.Button(nav, text="Tiếp →", command=self.next_row).pack(side="left")
        ttk.Button(nav, text="Tới mẫu chưa đánh giá", command=self.go_to_unreviewed).pack(side="left", padx=(18, 0))

        ttk.Label(
            nav,
            text="←/→: chuyển mẫu | Space: phát",
        ).pack(side="right")

    def _resolve_audio_path(self, audio_path: str) -> Path:
        """Đổi đường dẫn tương đối trong label.csv thành đường dẫn thật."""
        return (self.label_path.parent / audio_path).resolve()

    def _load_current_row(self) -> None:
        """Hiển thị mẫu hiện tại lên giao diện."""
        self.stop_audio()
        row = self.rows[self.index]

        self.progress_var.set(f"Mẫu {self.index + 1} / {len(self.rows)}")
        self.audio_var.set(row["audio_path"])
        self.label_status_var.set(row["label_status"])
        self.audio_quality_var.set(row["audio_quality"])

        self.label_text.configure(state="normal")
        self.label_text.delete("1.0", "end")
        self.label_text.insert("1.0", row["label"])
        self.label_text.configure(state="disabled")

        self.note_text.delete("1.0", "end")
        self.note_text.insert("1.0", row["note"])

        self._update_summary()

        # Tự động phát audio sau khi chuyển sang mẫu mới.
        self.root.after(100, self.play_audio)

    def _save_current_row(self) -> None:
        """Lưu đánh giá mẫu hiện tại vào review.csv."""
        row = self.rows[self.index]
        row["label_status"] = self.label_status_var.get()
        row["audio_quality"] = self.audio_quality_var.get()
        row["note"] = self.note_text.get("1.0", "end-1c").strip()

        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        with self.review_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        self._update_summary()

    def _update_summary(self) -> None:
        """Cập nhật thống kê nhanh trên giao diện."""
        correct = sum(row["label_status"] == "correct" for row in self.rows)
        wrong = sum(row["label_status"] == "wrong" for row in self.rows)
        uncertain = sum(row["label_status"] == "uncertain" for row in self.rows)
        easy = sum(row["audio_quality"] == "easy" for row in self.rows)
        hard = sum(row["audio_quality"] == "hard" for row in self.rows)

        reviewed = sum(
            bool(row["label_status"] and row["audio_quality"])
            for row in self.rows
        )

        self.summary_var.set(
            f"Đã đánh giá đủ: {reviewed}/{len(self.rows)} | "
            f"Đúng: {correct} | Sai: {wrong} | Không chắc: {uncertain} | "
            f"Dễ nghe: {easy} | Khó nghe: {hard}"
        )

    def play_audio(self) -> None:
        """Phát file WAV của mẫu hiện tại."""
        if winsound is None:
            messagebox.showerror(
                "Không hỗ trợ",
                "Phát audio bằng winsound chỉ hỗ trợ Windows.",
            )
            return

        audio_path = self._resolve_audio_path(self.rows[self.index]["audio_path"])
        if not audio_path.exists():
            messagebox.showerror("Thiếu audio", f"Không tìm thấy:\n{audio_path}")
            return

        if audio_path.suffix.lower() != ".wav":
            messagebox.showerror(
                "Định dạng chưa hỗ trợ",
                f"Hiện tool phát trực tiếp file WAV.\nFile hiện tại: {audio_path.name}",
            )
            return

        winsound.PlaySound(
            str(audio_path),
            winsound.SND_FILENAME | winsound.SND_ASYNC,
        )

    def stop_audio(self) -> None:
        """Dừng audio đang phát."""
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)

    def prev_row(self) -> None:
        """Lưu và chuyển về mẫu trước."""
        self._save_current_row()
        if self.index > 0:
            self.index -= 1
            self._load_current_row()

    def next_row(self) -> None:
        """Lưu và chuyển sang mẫu tiếp theo."""
        self._save_current_row()
        if self.index < len(self.rows) - 1:
            self.index += 1
            self._load_current_row()
        else:
            messagebox.showinfo("Hoàn thành", "Đã tới mẫu cuối cùng.")

    def go_to_unreviewed(self) -> None:
        """Chuyển tới mẫu đầu tiên chưa đánh giá đủ."""
        self._save_current_row()
        for i, row in enumerate(self.rows):
            if not row["label_status"] or not row["audio_quality"]:
                self.index = i
                self._load_current_row()
                return

        messagebox.showinfo("Hoàn thành", "Tất cả mẫu đã được đánh giá đủ.")

    def _on_close(self) -> None:
        """Lưu dữ liệu trước khi đóng cửa sổ."""
        self._save_current_row()
        self.stop_audio()
        self.root.destroy()


def main() -> None:
    """Mở công cụ review label cho một dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Tên dataset, ví dụ: VIVOS.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    processed_dir = PROJECT_ROOT / "data" / "processed" / dataset_name
    label_path = processed_dir / "label.csv"
    review_path = processed_dir / "review.csv"

    try:
        root = tk.Tk()
        ReviewApp(root, label_path, review_path)
        root.mainloop()
    except (FileNotFoundError, ValueError) as exc:
        if "root" in locals():
            root.destroy()
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

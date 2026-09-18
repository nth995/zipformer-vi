# zipformer_vi

Dự án chuẩn bị dữ liệu, huấn luyện và đánh giá Zipformer cho tiếng Việt.

## Cấu trúc

- `data/raw/`: dữ liệu gốc của từng bộ dữ liệu.
- `data/processed/`: file label và dữ liệu đã chuẩn hóa của từng bộ dữ liệu.
- `dataset_scripts/`: code xử lý riêng cho từng bộ dữ liệu.
- `run/`: các script chạy tiện ích như tạo label, kiểm tra label, gộp label.
- `training/`: code huấn luyện Zipformer.
- `evaluation/`: code đánh giá mô hình.
- `checkpoints/`: lưu checkpoint mô hình.

Hiện tại đây chỉ là cấu trúc khung. Các file xử lý cụ thể sẽ được thêm dần theo từng bộ dữ liệu.

## Datasets đã hỗ trợ

- VIVOS: tạo `label.csv` từ cấu trúc gốc.

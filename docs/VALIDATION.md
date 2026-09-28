# Kết quả kiểm tra

Đã kiểm tra trên Windows, Python 3.12.14. Phiên bản dependency thực tế được lưu trong `requirements-lock.txt`.

## Kết quả

- `python -m pytest -q`: **68 passed** (cập nhật biểu đồ toàn bộ dữ liệu, ngày 2026-09-28).
- `python -m ruff check .`: **All checks passed**.
- `python -m ruff format --check .`: **24 files already formatted**.
- `python -m pip check`: **No broken requirements found**.
- CLI với `examples/sales_dirty.csv` và pipeline mẫu: **21 → 20 dòng**, tạo đủ bộ đầu ra tại `outputs/demo`.
- CLI với JSON lồng tại nhánh `data.records`: **3 → 3 dòng**, tạo đầu ra tại `outputs/json_demo`.
- Streamlit AppTest: nạp mẫu → nhập pipeline 7 bước → xem trước → áp dụng → tạo HTML/ZIP; sửa pipeline loại bỏ kết quả và artifact cũ.
- Trình duyệt localhost: giao diện khởi động, nạp 21 dòng mẫu, tab EDA và biểu đồ Plotly hiển thị.
- ML: kiểm tra fit chỉ trên train bằng cách thay đổi toàn bộ giá trị test và xác nhận tham số/matrix train không đổi; category chỉ có ở test không xuất hiện trong schema đã học.
- ML: kiểm tra Standard/MinMax/Robust, điền thiếu, cột hằng/toàn thiếu, phân tầng, thứ tự thời gian, target thiếu, mã hóa nhãn và nạp lại joblib để transform dữ liệu mới.
- CLI ML mẫu: **12 train / 4 validation / 5 test, 6 feature**, tạo `outputs/ml_demo/ml_ready.zip`.
- Trình duyệt: tab Học máy chạy thành công, hiển thị train/validation/test, biểu đồ trước–sau và nút tải ZIP.
- `pip install --dry-run -r requirements.txt`: phân giải cấu hình triển khai thành công trên môi trường Windows hiện tại.

## Các trường hợp quan trọng

Bảo toàn mã `001`; ký hiệu thiếu cấu hình rõ ràng; CSV lỗi không bị bỏ qua; JSON làm phẳng không mất dữ liệu do trùng tên; SQLite được mở chỉ đọc và đóng kết nối trước dọn file tạm trên Windows; định dạng ngày xác định; lỗi chuyển kiểu có chính sách rõ ràng; không sửa bản gốc khi pipeline thất bại; cột toàn thiếu không bị tự tạo mean/median/mode; histogram có giới hạn bin trước cấp phát; so sánh dùng cùng bin; HTML escape nội dung nguồn và nhúng Plotly.js một lần; CLI chạy được khi đường dẫn tiếng Việt và terminal mặc định cp1252.

## Giới hạn của việc kiểm tra

Đã kiểm tra chức năng và luồng chính trên fixture nhỏ, kèm benchmark 2 triệu dòng bên dưới. Không khẳng định SLA cho mọi bảng 2 triệu dòng, mọi hệ điều hành hoặc mọi phiên bản thư viện trong khoảng dependency cho phép; dùng lock file để tái tạo môi trường đã kiểm tra.

Workflow GitHub Actions cho Ubuntu/Windows đã được viết nhưng chưa chạy trên GitHub trong lượt cập nhật này. Chưa triển khai production trên Streamlit Community Cloud.

## Benchmark 2 triệu dòng — 2026-09-28

Chạy `python -X utf8 scripts/benchmark_large.py --ml` trên Windows 11, Python 3.12.14, pandas 3.0.6. Máy có khoảng 15,8 GiB RAM; đây là số đo một lần chạy, không phải phép so sánh hiệu năng giữa các phiên bản.

Dữ liệu tổng hợp: **2.000.000 dòng × 5 cột**, gồm mã có số 0 đầu, 2 feature số, 1 feature phân loại và target số. Có 2.000 ô thiếu, 500.000 chuỗi cần trim. File CSV 58.650.438 byte (~55,9 MiB); bảng sau nạp ~104,9 MiB.

| Công đoạn | Thời gian |
|---|---:|
| Tạo CSV | 4,234 giây |
| Nạp và kiểm tra | 8,152 giây |
| Thống kê toàn bộ bảng | 5,662 giây |
| Điền median, trim, ghi nhật ký | 13,092 giây |
| Tạo HTML và ZIP chứa CSV đầy đủ | 19,716 giây |
| Đọc lại CSV trong ZIP và đếm bản ghi | 1,929 giây |
| Chia tập, điền thiếu, StandardScaler, one-hot | 7,088 giây |

- Xác nhận CSV trong ZIP có đúng **2.000.000 bản ghi**; bản gốc còn nguyên giá trị thiếu và mã `000000000`.
- Nhật ký đếm đúng số ô thay đổi qua từng bước; ZIP ~15,5 MiB.
- ML tạo **1.200.000 train / 400.000 validation / 400.000 test**, mỗi tập có 6 feature, ma trận hữu hạn. Không huấn luyện mô hình hoặc benchmark xuất gói ML trong lần này.
- RAM đỉnh của tiến trình: **955,4 MiB**, tính cả interpreter/thư viện. Đây là peak working set, không phải tổng RAM hệ thống hay mức RAM bảo đảm cho triển khai.
- Kết quả máy đọc được: `outputs/benchmark_2m/results.json`. Script benchmark được lưu trong Git; dữ liệu và artifact sinh ra được bỏ qua bởi `.gitignore`.

Benchmark dùng đường dẫn local, chưa bao gồm upload trình duyệt, nhiều phiên đồng thời, bảng rộng 200 cột, JSON lồng lớn hoặc giới hạn tài nguyên Community Cloud. Unit test/UI test kiểm tra thêm đọc JSONL nhiều lô, lỗi cuối file, chặn vượt giới hạn, bảo toàn bản gốc khi dùng snapshot nhẹ và cache thống kê. Sau benchmark này, cấu hình biểu đồ đã đổi sang dùng toàn bộ dòng/cột, không lấy mẫu; cần chạy lại phép đo trước khi so sánh hiệu năng report mới.

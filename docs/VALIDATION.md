# Kết quả kiểm tra

Đã kiểm tra trên Windows, Python 3.12.14. Phiên bản dependency thực tế được lưu trong `requirements-lock.txt`.

## Kết quả

- `python -m pytest -q`: **59 passed** (bản 1.1, ngày 2026-09-19).
- `python -m ruff check .`: **All checks passed**.
- `python -m ruff format --check .`: **22 files already formatted**.
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

Đã kiểm tra chức năng và luồng chính trên fixture nhỏ, không khẳng định SLA hiệu năng ở giới hạn tối đa 200.000 dòng. Không kiểm tra trên mọi hệ điều hành hoặc mọi phiên bản thư viện trong khoảng dependency cho phép; dùng lock file để tái tạo môi trường đã kiểm tra.

Workflow GitHub Actions cho Ubuntu/Windows đã được viết nhưng chưa chạy trên GitHub trong lượt cập nhật này. Chưa triển khai production trên Streamlit Community Cloud.

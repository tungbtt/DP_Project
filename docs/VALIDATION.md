# Kết quả kiểm tra

Đã kiểm tra trên Windows, Python 3.12.14. Phiên bản dependency thực tế được lưu trong `requirements-lock.txt`.

## Kết quả

- `python -m pytest -q`: **32 passed**.
- `python -m ruff check .`: **All checks passed**.
- `python -m ruff format --check .`: **16 files already formatted**.
- `python -m pip check`: **No broken requirements found**.
- CLI với `examples/sales_dirty.csv` và pipeline mẫu: **21 → 20 dòng**, tạo đủ bộ đầu ra tại `outputs/demo`.
- CLI với JSON lồng tại nhánh `data.records`: **3 → 3 dòng**, tạo đầu ra tại `outputs/json_demo`.
- Streamlit AppTest: nạp mẫu → nhập pipeline 7 bước → xem trước → áp dụng → tạo HTML/ZIP; sửa pipeline loại bỏ kết quả và artifact cũ.
- Trình duyệt localhost: giao diện khởi động, nạp 21 dòng mẫu, tab EDA và biểu đồ Plotly hiển thị.

## Các trường hợp quan trọng

Bảo toàn mã `001`; ký hiệu thiếu cấu hình rõ ràng; CSV lỗi không bị bỏ qua; JSON làm phẳng không mất dữ liệu do trùng tên; SQLite được mở chỉ đọc và đóng kết nối trước dọn file tạm trên Windows; định dạng ngày xác định; lỗi chuyển kiểu có chính sách rõ ràng; không sửa bản gốc khi pipeline thất bại; cột toàn thiếu không bị tự tạo mean/median/mode; histogram có giới hạn bin trước cấp phát; so sánh dùng cùng bin; HTML escape nội dung nguồn và nhúng Plotly.js một lần; CLI chạy được khi đường dẫn tiếng Việt và terminal mặc định cp1252.

## Giới hạn của việc kiểm tra

Đã kiểm tra chức năng và luồng chính trên fixture nhỏ, không khẳng định SLA hiệu năng ở giới hạn tối đa 200.000 dòng. Không kiểm tra trên mọi hệ điều hành hoặc mọi phiên bản thư viện trong khoảng dependency cho phép; dùng lock file để tái tạo môi trường đã kiểm tra.

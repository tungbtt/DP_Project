# Phạm vi đồ án và nghiệm thu

## Mục tiêu

Thư viện Python và giao diện giúp khám phá, đánh giá chất lượng, tiền xử lý dữ liệu dạng bảng và tạo báo cáo HTML offline. Không phụ thuộc một bộ dữ liệu cố định.

## Use cases đã triển khai

1. Nạp dữ liệu CSV/TSV/JSON/JSONL/XLSX/SQLite; Parquet tùy chọn.
2. Xem trước cấu trúc, chọn vai trò cột và cấu hình đọc.
3. Thống kê mô tả, trực quan phân phối, tương quan và scatter.
4. Phát hiện thiếu, trùng, cột hằng, khoảng trắng, sai kiểu số/ngày và ngoại lệ.
5. Đề xuất xử lý dựa trên quy tắc, người dùng quyết định từng biến đổi.
6. Tạo/sắp xếp/xóa bước pipeline; xem trước đầy đủ trên bản sao.
7. Áp dụng pipeline, kiểm tra tác động và vấn đề còn tồn tại.
8. Xuất CSV, HTML, ZIP, schema, metadata, nhật ký và pipeline tái sử dụng.
9. Chuẩn bị dữ liệu học máy: chọn target/features; chia ngẫu nhiên/phân tầng/thời gian; điền thiếu, chuẩn hóa và mã hóa chỉ fit trên train; xuất các tập và fitted preprocessor.
10. Nhập link dataset Kaggle hoặc file Google Drive công khai, tải và chọn file trong ZIP, cấu hình đọc rồi nạp qua bộ đọc chung. Kaggle hỗ trợ token hoặc username/API key cũ; Drive chưa có OAuth cho file riêng tư.
11. Điều hướng 5 bước, chỉ tạo biểu đồ ở màn hình đang mở; giữ dữ liệu/pipeline/kết quả khi đổi bước và cấu hình ML khi rời trang. Nạp bộ dữ liệu mới vô hiệu hóa kết quả của bộ trước.

## Nguyên tắc chất lượng

- Không âm thầm bỏ bản ghi CSV sai cấu trúc.
- Không sửa nguồn. Thất bại giữa quy trình không trả về dữ liệu thành công một phần.
- Không tự coi ID là đại lượng đo hoặc xóa ngoại lệ.
- So sánh trước–sau ghi rõ số dòng/cột và mẫu số tỷ lệ thiếu.
- Thống kê và biểu đồ dùng toàn bộ dữ liệu hợp lệ đã nạp, không lấy mẫu.
- Đầu ra HTML nhúng thư viện để xem offline; CLI và UI dùng cùng lõi.

## Kịch bản bảo vệ

Nạp `examples/sales_dirty.csv`, kiểm tra ID `001`, tỷ lệ thiếu, thu nhập `unknown`, tuổi 150 và một bản sao dư. Nhập pipeline mẫu, xem log từng bước, áp dụng. Kết quả có 20 dòng; tuổi nằm trong miền đã cấu hình; thu nhập hữu hạn và được clip theo IQR; ngày không hợp lệ vẫn thiếu và được báo cáo. Xuất HTML, ngắt mạng và mở báo cáo. Chạy thêm `examples/records.json` với nhánh `data.records` để chứng minh đầu vào khác cấu trúc.

## Mở rộng sau

Kết nối SQL ngoài SQLite; che dữ liệu nhạy cảm; rule nhiều cột; dữ liệu lớn theo chunk; so sánh các file theo thời gian; chia ML theo nhóm và huấn luyện mô hình. Các tính năng này không nằm trong bản hiện tại.

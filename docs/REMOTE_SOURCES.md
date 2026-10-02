# Nhập dữ liệu từ Kaggle và Google Drive

Chức năng ở bước **01 · Nguồn dữ liệu** của Data Prep Studio 1.3. Sau khi nạp, dữ liệu dùng chung luồng EDA, làm sạch, báo cáo và chuẩn bị ML như file từ máy tính.

## Kaggle

1. Chọn **Kaggle**, dán link `https://www.kaggle.com/datasets/owner/name`.
2. Nhấn **Tải danh sách file**. Link `/versions/2` được hỗ trợ; tham số `?select=...` không tự chọn file.
3. Nếu dataset yêu cầu xác thực, mở **Xác thực Kaggle**, nhập API token từ Settings → API. Có thể dùng username + API key cũ thay token. Token được ưu tiên khi nhập cả hai cách.
4. Chọn file trong ZIP. Ứng dụng chỉ giải nén và nạp file đã chọn, không tự nối nhiều bảng.
5. Mở tùy chọn đọc nếu cần rồi **Nạp dữ liệu**.

Dataset phải cho phép tải và tài khoản phải có quyền; công cụ không chấp nhận thay điều khoản Kaggle. Link competition/notebook chưa hỗ trợ. Một số dataset công khai vẫn có thể yêu cầu đăng nhập hoặc gặp giới hạn tải của dịch vụ.

## Google Drive

1. Trong Drive, chia sẻ một file bằng **Bất kỳ ai có đường liên kết**, cho phép tải xuống.
2. Chọn **Google Drive**, dán link dạng `https://drive.google.com/file/d/FILE_ID/view` hoặc `/open?id=FILE_ID`, `/uc?id=FILE_ID`.
3. Nhấn **Tải danh sách file**, chọn file nếu nguồn là ZIP, rồi **Nạp dữ liệu**.

Ứng dụng xử lý trang xác nhận tải file lớn theo biểu mẫu GET của Drive. File cần đăng nhập, thư mục, Google Sheets/Docs chưa hỗ trợ. Với Sheets, xuất CSV/XLSX rồi chia sẻ file xuất. Hạn mức hoặc giới hạn quyền tải của Drive vẫn áp dụng. Không đọc cookie trình duyệt, không có đăng nhập OAuth.

## Định dạng và giới hạn

- CSV/TSV, JSON/JSONL/NDJSON, XLSX, SQLite; Parquet nếu đã cài dependency tùy chọn.
- Giới hạn gói tải 512 MiB, file được chọn sau giải nén 512 MiB, tối đa 2.000.000 dòng và 200 cột. ZIP tối đa 1.000 mục; file lớn hơn giới hạn không xuất hiện trong danh sách chọn. Không hỗ trợ ZIP lồng hoặc ZIP mã hóa.
- Không tự lấy mẫu hay bỏ dòng. Khi vượt giới hạn hoặc file lỗi, báo lỗi và giữ bộ dữ liệu đang làm việc.
- Tải theo luồng xuống đĩa tạm, có giới hạn byte ngay cả khi máy chủ không cung cấp Content-Length. Thời gian tải cho phép 300 giây, timeout kết nối 15 giây / đọc 30 giây (thời gian thực tế có thể vượt mốc 300 trong lúc chờ một lần đọc mạng).
- Cùng tùy chọn đọc local: encoding, delimiter, dấu thập phân, ký hiệu thiếu, nhánh JSON, sheet Excel, bảng SQLite. Khi máy chủ không trả tên file, nhập tên có đuôi đúng ở **Tên file và định dạng**, ví dụ `data.csv`.
- ZIP không giải nén theo đường dẫn do nguồn cung cấp; từ chối đường dẫn tuyệt đối/thoát thư mục, symlink, tên trùng. File tải xuống chỉ từ HTTPS của nhà cung cấp được cho phép; chặn địa chỉ IP nội bộ trong kết quả DNS.

## Trạng thái và thông tin nguồn

File tải tạm thuộc từng phiên, không dùng cache chia sẻ giữa người dùng. Tải thành công nguồn mới hoặc đặt lại phiên sẽ dọn file tạm trước đó. Đối tượng thư mục tạm được dọn khi thu hồi; nếu tiến trình bị tắt đột ngột, hệ điều hành có thể còn file tạm cần dọn. Đổi màn hình giữ nguồn đã tải; đổi link cần tải lại trước khi nạp.

Ô token/key được xóa sau mỗi lần tải. Không ghi credential hay URL chuyển hướng có chữ ký vào metadata hoặc thông báo lỗi. `source_metadata.json` ghi nhà cung cấp, link nguồn chuẩn hóa và tên file được chọn; resource key của Drive không được đưa vào báo cáo. Tên/link nguồn vẫn là thông tin trong kết quả xuất, nên kiểm tra trước khi chia sẻ báo cáo.

## API Python

```python
from dataprep.remote import fetch_remote
from dataprep.io import LoadOptions

download = fetch_remote("https://www.kaggle.com/datasets/uciml/iris")
try:
    print(download.files())
    dataset = download.load("Iris.csv", LoadOptions())
    print(dataset.frame.shape)
finally:
    download.close()
```

CLI `python -m dataprep` vẫn nhận đường dẫn local. Nhập link nằm trong giao diện web và module Python `dataprep.remote`.

## Nguồn kỹ thuật

- [Kaggle API chính thức](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md): dataset download và các cách xác thực.
- [Google Drive download](https://developers.google.com/workspace/drive/api/guides/manage-downloads): quyền tải và các loại tài liệu.
- [gdown download implementation](https://github.com/wkentaro/gdown/blob/main/gdown/download.py): tham khảo cơ chế biểu mẫu xác nhận của link công khai. Ứng dụng dùng requests trực tiếp, không phụ thuộc gdown.

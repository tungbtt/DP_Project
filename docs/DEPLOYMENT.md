# Deploy từ GitHub lên Streamlit Community Cloud

Repo hiện tại: `https://github.com/tungbtt/DP_Project`, nhánh hiện tại `master`, entrypoint `app.py`.

GitHub lưu mã nguồn. Streamlit Community Cloud chạy Python và cung cấp URL web. GitHub Pages chỉ phục vụ nội dung tĩnh, có thể đăng `report.html` nhưng không chạy giao diện xử lý dữ liệu Python.

## Các bước triển khai

1. Commit/push các thay đổi đã kiểm tra lên repo GitHub (bao gồm module ML, dependency và workflow).
2. Truy cập https://share.streamlit.io/ và đăng nhập, kết nối tài khoản GitHub của bạn.
3. Chọn **Create app** và triển khai từ repo có sẵn.
4. Chọn repository **tungbtt/DP_Project**, branch **master**, main file path **app.py**.
5. Trong **Advanced settings**, chọn **Python 3.12** để khớp môi trường đã kiểm tra.
6. Chọn tên miền phụ nếu muốn và nhấn **Deploy**. Khi build thành công, Cloud cung cấp URL `*.streamlit.app`.
7. Mở ứng dụng, dùng dữ liệu mẫu và chạy thử tab EDA, Học máy, tải HTML/ZIP.

Các bước kết nối tài khoản và deploy cần được thực hiện trong tài khoản Streamlit/GitHub của chủ repo. Không có URL production cho đến khi deploy thành công.

## Cấu hình đã có trong project

- `requirements.txt`: cài project Python cùng dependency khai báo trong `pyproject.toml`, khóa phiên bản bằng constraints từ `requirements-lock.txt`.
- `.streamlit/config.toml`: theme, upload tối đa 100 MiB, tắt thống kê sử dụng Streamlit.
- `.github/workflows/tests.yml`: chạy test/lint trên Ubuntu và Windows, Python 3.12 khi push/PR.
- `.gitignore`: loại `.venv`, outputs và secrets khỏi Git; không đưa dữ liệu người dùng vào repo.
- `requirements-lock.txt`: phiên bản đầy đủ đã kiểm tra, dùng cho CI/tái lập local; Cloud dùng file này làm constraints để giữ đúng phiên bản dependency của ứng dụng mà không cần cài công cụ dev.

Không cần cấu hình secret cho chức năng hiện tại. Khi mở rộng kết nối DB, đặt credential trong phần Secrets của Cloud, không commit vào Git.

## Vận hành

App xử lý dữ liệu trong RAM của máy chủ, không upload file của người dùng vào GitHub. Kết quả được tải về bằng nút download. Làm mới/đóng phiên hoặc restart app có thể làm mất trạng thái chưa tải xuống.

Khi dùng web, file được gửi từ trình duyệt đến máy chủ Streamlit; nếu dữ liệu cần giữ hoàn toàn trên máy cá nhân, dùng bản local. Giới hạn 100 MiB/200.000 dòng trong app không phải cam kết máy chủ Cloud đủ RAM cho mọi file. Nên thử dữ liệu nhỏ trước và điều chỉnh `MAX_BYTES`, `MAX_ROWS`, `MAX_COLUMNS` trong `dataprep/io.py` cùng `server.maxUploadSize` nếu cần.

GitHub Actions kiểm tra code; workflow không tự tạo ứng dụng Cloud và không phải cổng chặn việc Cloud tự cập nhật từ nhánh đang deploy. Muốn kiểm soát phát hành, chỉ merge vào nhánh deploy sau khi CI đạt.

## Nguồn chính thức

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages

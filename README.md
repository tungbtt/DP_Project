# Data Prep Studio

Đồ án Python: tự động khám phá dữ liệu dạng bảng, làm sạch có giải thích và xuất báo cáo HTML tương tác bằng Plotly.

Phiên bản 1.1 bổ sung tab **Học máy**: chia train/validation/test, chuẩn hóa, mã hóa và xuất bộ tiền xử lý đã fit. Xem [hướng dẫn ML](docs/MACHINE_LEARNING.md) và [deploy từ GitHub lên Streamlit Cloud](docs/DEPLOYMENT.md).

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên (khuyến nghị 3.11/3.12). Mở terminal tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Sau đó mở `http://localhost:8501`. Không cần kích hoạt môi trường để chạy các lệnh trên. Có thể dùng `scripts/run.ps1` sau khi cài đặt.

Để dùng đúng phiên bản thư viện đã kiểm tra, cài `requirements-lock.txt` trước bước cài project:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 -m pip install -e .
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Parquet là tùy chọn: `python -m pip install -e ".[parquet]"` trong môi trường đang dùng.

## Quy trình sử dụng

1. Nạp file hoặc nhấn **Dùng dữ liệu mẫu**. Điều chỉnh dấu phân cách, encoding, ký hiệu thiếu khi cần.
2. Xác nhận vai trò cột trong tab **Dữ liệu**. Vai trò không tự đổi kiểu dữ liệu; ID được loại khỏi tương quan.
3. Xem thống kê, vấn đề và biểu đồ ở **Khám phá EDA**.
4. Thêm các bước ở **Làm sạch**, hoặc nạp `examples/pipeline.json` khi dùng dữ liệu mẫu.
5. Nhấn **Xem trước toàn bộ quy trình**, kiểm tra dữ liệu và nhật ký, rồi **Áp dụng kết quả đã xem trước**.
6. Xem **So sánh**, sau đó tạo và tải báo cáo ở **Xuất kết quả**.
7. Nếu chuẩn bị cho mô hình, mở **6. Học máy**, chọn target/features, phương pháp chuẩn hóa và chia tập, rồi tải gói ML. Luồng này dùng dữ liệu gốc; không dùng thống kê đã fit toàn bộ dữ liệu ở tab Làm sạch.

Sửa pipeline hoặc vai trò cột sẽ vô hiệu hóa kết quả cũ để tránh xuất báo cáo sai cấu hình. Mỗi lần chạy đều bắt đầu từ bản gốc. Nếu một bước lỗi, không xuất kết quả chạy dở. Đóng phiên làm việc sẽ mất dữ liệu đang giữ trong bộ nhớ; hãy tải ZIP/pipeline để lưu lại.

## Chạy bằng dòng lệnh

```powershell
.\.venv\Scripts\python.exe -m dataprep examples/sales_dirty.csv --pipeline examples/pipeline.json --output outputs/demo
.\.venv\Scripts\python.exe -m dataprep examples/records.json --json-path data.records --output outputs/json_demo
```

Bỏ `--pipeline` để chỉ chạy EDA. Thêm `--table ten_bang` cho SQLite nhiều bảng. Dùng `--delimiter ";" --decimal ","` cho dữ liệu phù hợp. `--missing-token NA` là tùy chọn lặp được. CLI từ chối ghi đè đầu ra có sẵn trừ khi có `--overwrite`.

Đầu ra: `report.html`, `cleaned_data.csv`, `pipeline.json`, `processing_log.json`, `schema.json`, `source_metadata.json`, `result.zip`.

## Chức năng

- CSV/TSV, JSON/JSONL, XLSX, SQLite; Parquet khi cài dependency tùy chọn.
- Phát hiện dấu phân cách CSV; kiểm tra dòng sai số trường, tiêu đề trùng và dữ liệu rỗng.
- JSON object lồng nhau được làm phẳng bằng dấu chấm; mảng lồng giữ thành chuỗi JSON, không tự nhân số dòng. Nhánh JSON dùng đường dẫn dấu chấm.
- Thống kê số, phân loại, chuỗi, ngày giờ; thiếu, bản sao dư, cột hằng, sai kiểu và ngoại lệ IQR.
- Histogram, boxplot, top-category bar, thiếu theo cột, tương quan Pearson/Spearman, scatter, số bản ghi theo ngày.
- Chuẩn hóa chuỗi, ánh xạ nhãn, chuyển số/ngày, điền thiếu, xóa dòng thiếu, loại trùng, xóa cột, IQR và miền giá trị.
- Nhật ký từng bước; so sánh trước–sau; pipeline JSON có thể sửa, sắp xếp và tái sử dụng.
- Báo cáo HTML offline nhúng Plotly.js một lần; nội dung văn bản nguồn được escape trong template.
- Chuẩn bị ML: StandardScaler/MinMaxScaler/RobustScaler; one-hot/ordinal; train/validation/test; imputer/encoder/scaler fit chỉ trên train, dùng lại cho dữ liệu mới.

## Kiến trúc

```text
app.py                 Giao diện Streamlit, trạng thái phiên và xem trước
dataprep/io.py         Đọc và kiểm tra đầu vào
dataprep/profile.py    Thống kê, vai trò cột, cảnh báo và tương quan
dataprep/pipeline.py   Biến đổi có thứ tự, validation, nhật ký
dataprep/charts.py     Biểu đồ Plotly trên toàn bộ dữ liệu hợp lệ
dataprep/report.py     HTML và ZIP
dataprep/cli.py        Giao diện dòng lệnh dùng cùng lõi xử lý
dataprep/ml.py         Chia tập, chuẩn hóa, mã hóa và artifact cho ML
dataprep/ml_ui.py      Giao diện cấu hình dữ liệu học máy
dataprep/templates/    Mẫu báo cáo HTML
examples/             Dữ liệu lỗi chủ động và pipeline minh họa
tests/                Kiểm tra dữ liệu, CLI và luồng Streamlit
docs/                 Đặc tả và định dạng pipeline
```

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Kiểm tra bao gồm bảo toàn ID, dữ liệu lỗi, JSON lồng, SQLite chỉ đọc, Excel, công thức thống kê, quy trình thất bại nguyên tử, kết quả tái lập, báo cáo offline, CLI và luồng UI mẫu → xem trước → áp dụng → xuất.

## Giới hạn có chủ đích

- Dữ liệu dạng bảng, tối đa **512 MiB / 2.000.000 dòng / 200 cột**. Đây là giới hạn đầu vào, không phải cam kết hiệu năng trên mọi cấu hình máy. Toàn bộ bảng được giữ trong RAM; bảng rộng hoặc nhiều chuỗi cần nhiều bộ nhớ hơn bảng số hẹp. Excel vẫn chịu giới hạn số dòng của định dạng XLSX; dùng CSV/JSONL/Parquet/SQLite cho 2 triệu dòng.
- SQL hiện hỗ trợ file SQLite chỉ đọc; không thực thi file `.sql`, không kết nối MySQL/PostgreSQL.
- Không tự suy luận định dạng ngày. Chuyển ngày yêu cầu format, kết quả chuẩn hóa timezone UTC. Khi dữ liệu nghiệp vụ cần timezone địa phương, phải chuẩn hóa nguồn trước.
- Ký hiệu thiếu mặc định chỉ là ô rỗng; `NA`, `null`, `0` không tự coi là thiếu. Khoảng trắng chỉ trở thành thiếu khi chọn bước trim.
- Không tự hiểu mọi nghiệp vụ hoặc tự xóa ngoại lệ. IQR bằng 0 / dưới 4 giá trị số thì không đưa ra ngưỡng.
- Mọi biểu đồ dùng toàn bộ dữ liệu hợp lệ, không lấy mẫu: tương quan dùng toàn bộ dòng và cột số, scatter dùng toàn bộ cặp và WebGL, phân phối được tạo cho mọi cột không phải ID/ignore, so sánh được tạo cho mọi cột số. Histogram, boxplot và biểu đồ phân loại tổng hợp toàn bộ bản ghi thay vì vẽ mỗi bản ghi thành một điểm. Báo cáo có thể tạo chậm và trình duyệt cần nhiều RAM với bảng lớn.
- CSV không lưu đầy đủ dtype; `schema.json` để đối chiếu. Giá trị thiếu và chuỗi rỗng có thể không phân biệt được khi xuất CSV.
- Report có thể chứa giá trị dữ liệu trong nhãn/tần suất, chưa có tính năng che dữ liệu nhạy cảm.
- Đã chuẩn bị dữ liệu ML có giám sát; chưa huấn luyện/tối ưu mô hình, chia theo nhóm, tự học luật nghiệp vụ, AI/LLM hoặc cơ sở dữ liệu lưu lịch sử.

## Dữ liệu lớn và kiểm tra 2 triệu dòng

Đọc CSV từ đường dẫn không tạo thêm bản sao bytes/chuỗi của cả file; JSONL đọc từng lô 50.000 bản ghi. Nhật ký thay đổi so sánh từng cột/lô, và CSV được ghi từng lô trực tiếp vào ZIP. Giao diện giữ thống kê trong phiên để không tính lại mỗi lần đổi widget. Với hơn 200.000 dòng, lấy CSV đầy đủ trong ZIP để tránh giữ thêm bản tải CSV trong RAM. Biểu đồ không lấy mẫu theo yêu cầu hiện tại nên có thể trở thành phần tốn thời gian và RAM nhất.

Đây vẫn là xử lý trong RAM, chưa phải hệ thống out-of-core. JSON dạng mảng và Parquet vẫn nạp cả bảng; upload web, dữ liệu gốc, dữ liệu sạch, ma trận ML và gói ZIP có thể đồng thời chiếm bộ nhớ. Tăng dung lượng upload không tăng RAM của hosting. Để chạy thường xuyên trên dữ liệu lớn, dùng CLI/local hoặc máy chủ có RAM phù hợp sau khi đo bằng dữ liệu thực.

Benchmark tùy chọn (tạo dữ liệu tổng hợp 2 triệu dòng, nạp, thống kê, làm sạch, HTML/ZIP và kiểm chứng số bản ghi xuất):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_large.py
# Kiểm tra thêm chuẩn bị dữ liệu ML:
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_large.py --ml
```

Kết quả và dữ liệu thử nằm trong `outputs/benchmark_2m/` (không commit). Xem [VALIDATION.md](docs/VALIDATION.md) cho kết quả đo và phạm vi kiểm tra.

## Triển khai web từ GitHub

Ứng dụng có thể chạy trên Streamlit Community Cloud từ repo `tungbtt/DP_Project`, branch `master`, file `app.py`, Python 3.12. GitHub Pages chỉ đăng được báo cáo HTML tĩnh. Xem [DEPLOYMENT.md](docs/DEPLOYMENT.md) để triển khai bằng tài khoản của bạn; cấu hình GitHub Actions đã có để kiểm tra trên Ubuntu và Windows khi push/PR.

## Tài liệu tham khảo

- [pandas I/O](https://pandas.pydata.org/docs/reference/io.html)
- [Plotly HTML export](https://plotly.com/python/interactive-html-export/)
- [Plotly performance](https://plotly.com/python/performance/)
- [Streamlit Plotly charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)

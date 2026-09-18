# Data Prep Studio

Đồ án Python: tự động khám phá dữ liệu dạng bảng, làm sạch có giải thích và xuất báo cáo HTML tương tác bằng Plotly.

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên (khuyến nghị 3.11/3.12). Mở terminal tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Sau đó mở `http://localhost:8501`. Không cần kích hoạt môi trường để chạy các lệnh trên. Có thể dùng `scripts/run.ps1` sau khi cài đặt.

Để dùng đúng phiên bản thư viện đã kiểm tra, cài `requirements-lock.txt` trước bước cài project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
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

## Kiến trúc

```text
app.py                 Giao diện Streamlit, trạng thái phiên và xem trước
dataprep/io.py         Đọc và kiểm tra đầu vào
dataprep/profile.py    Thống kê, vai trò cột, cảnh báo và tương quan
dataprep/pipeline.py   Biến đổi có thứ tự, validation, nhật ký
dataprep/charts.py     Biểu đồ Plotly, tổng hợp/lấy mẫu có ghi rõ
dataprep/report.py     HTML và ZIP
dataprep/cli.py        Giao diện dòng lệnh dùng cùng lõi xử lý
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

- Dữ liệu dạng bảng, tối đa 100 MiB / 200.000 dòng / 200 cột. Đây là giới hạn đầu vào, không phải cam kết hiệu năng trên mọi cấu hình máy. Toàn bộ bảng được giữ trong RAM.
- SQL hiện hỗ trợ file SQLite chỉ đọc; không thực thi file `.sql`, không kết nối MySQL/PostgreSQL.
- Không tự suy luận định dạng ngày. Chuyển ngày yêu cầu format, kết quả chuẩn hóa timezone UTC. Khi dữ liệu nghiệp vụ cần timezone địa phương, phải chuẩn hóa nguồn trước.
- Ký hiệu thiếu mặc định chỉ là ô rỗng; `NA`, `null`, `0` không tự coi là thiếu. Khoảng trắng chỉ trở thành thiếu khi chọn bước trim.
- Không tự hiểu mọi nghiệp vụ hoặc tự xóa ngoại lệ. IQR bằng 0 / dưới 4 giá trị số thì không đưa ra ngưỡng.
- Biểu đồ phân phối được tổng hợp từ toàn bộ giá trị hợp lệ. Scatter lấy mẫu tối đa 5.000 cặp, seed=42. HTML chỉ vẽ phân phối 8 cột đầu phù hợp / phiên bản, tương quan 30 cột, so sánh 6 cột; thống kê vẫn có cho mọi cột.
- CSV không lưu đầy đủ dtype; `schema.json` để đối chiếu. Giá trị thiếu và chuỗi rỗng có thể không phân biệt được khi xuất CSV.
- Report có thể chứa giá trị dữ liệu trong nhãn/tần suất, chưa có tính năng che dữ liệu nhạy cảm.
- Chưa bao gồm mô hình ML, tự học luật nghiệp vụ, AI/LLM, server nhiều người dùng hoặc cơ sở dữ liệu lưu lịch sử.

## Tài liệu tham khảo

- [pandas I/O](https://pandas.pydata.org/docs/reference/io.html)
- [Plotly HTML export](https://plotly.com/python/interactive-html-export/)
- [Plotly performance](https://plotly.com/python/performance/)
- [Streamlit Plotly charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)

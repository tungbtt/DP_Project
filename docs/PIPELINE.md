# Pipeline v1

```json
{
  "version": 1,
  "expected_columns": ["id", "age"],
  "roles": {"id": "id", "age": "numeric"},
  "steps": [
    {"op": "range", "columns": ["age"], "min": 0, "max": 120, "action": "missing"},
    {"op": "fill_missing", "columns": ["age"], "strategy": "median"}
  ]
}
```

`expected_columns` là tùy chọn; nếu có phải khớp tên và thứ tự cột nguồn. `roles` có thể gồm numeric/category/text/datetime/id/ignore. Vai trò là metadata cho EDA, không thay cho chuyển kiểu. Các bước chạy tuần tự, không thực thi Python/SQL do người dùng nhập.

| op | Tham số |
|---|---|
| normalize_text | columns; case=keep/lower/upper/casefold; empty_as_missing=true |
| replace_values | columns; mapping object, khóa khớp biểu diễn chuỗi của giá trị |
| cast | columns; dtype=numeric/datetime/string; errors=raise/coerce; format; decimal; thousands |
| fill_missing | columns; strategy=mean/median/mode/constant; value khi constant |
| drop_missing | columns (rỗng = toàn bảng); how=any/all |
| drop_duplicates | columns (rỗng = toàn bảng); keep=first/last/false |
| drop_columns | columns; không được xóa toàn bộ cột |
| outliers | columns số; factor=1.5; method=clip/missing/drop |
| range | columns số; min và/hoặc max; action=missing/drop |

Mặc định `cast.errors=raise`. Khi chọn coerce, giá trị lỗi thành thiếu, thể hiện ở nhật ký. `format` ngày dùng cú pháp như `%d/%m/%Y` hoặc `ISO8601`, không chấp nhận mixed. Ngày được đưa về UTC.

Điền mean/median yêu cầu các giá trị không thiếu đều là số hữu hạn. Cột toàn thiếu phải dùng constant. Mode có nhiều giá trị đồng hạng dùng giá trị đầu trong kết quả sắp xếp mode của pandas. Xử lý IQR nhiều cột với method=drop chạy theo thứ tự columns và tính lại ngưỡng trên các dòng còn lại; nếu cần ngưỡng cố định, dùng các bước range với giới hạn cụ thể.

Nhật ký `changed_cells_retained` đếm thay đổi giá trị tại các dòng/cột còn tồn tại theo biểu diễn chuỗi (hai ô thiếu được coi là bằng nhau). Thay đổi dtype được ghi riêng. Dòng/cột bị xóa có trường riêng. Nhật ký không lưu toàn bộ giá trị trước/sau để tránh phình dung lượng; bản gốc được giữ trong phiên ứng dụng, không đưa vào ZIP.

Tái lập cần cùng dữ liệu nguồn, cấu hình đọc (`source_metadata.json`), pipeline và môi trường dependency. Timestamp báo cáo khác nhau giữa các lần xuất; dữ liệu kết quả và log biến đổi là xác định.


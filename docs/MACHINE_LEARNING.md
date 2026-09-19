# Chuẩn hóa và chuẩn bị dữ liệu học máy

## Luồng xử lý

Dữ liệu gốc đã nạp → chọn target/features → xử lý dòng thiếu target theo lựa chọn → chia train/validation/test → fit imputer/scaler/encoder trên train → transform validation/test → xuất artifact.

Tab Học máy không lấy kết quả đã làm sạch ở tab 3. Median/mean, ngưỡng IQR, các phép biến đổi học trên toàn bộ dữ liệu có thể gây leakage khi chia tập sau đó. Nếu nguồn đã được tiền xử lý thống kê từ trước, ứng dụng không thể tự phát hiện hoặc đảo ngược leakage đó.

## Phương pháp

| Chức năng | Lựa chọn |
|---|---|
| Bài toán | Hồi quy, phân loại |
| Feature số | Điền median/mean/0; StandardScaler, MinMaxScaler, RobustScaler hoặc giữ thang đo |
| Feature phân loại | Điền nhãn thiếu cố định; OneHotEncoder hoặc OrdinalEncoder |
| Nhãn phân loại | LabelEncoder fit trên train, xuất mapping; lớp mới ở test gây thông báo lỗi |
| Chia tập | Ngẫu nhiên có seed; phân tầng theo lớp; hoặc thứ tự thời gian |
| Giá trị số lỗi | Dừng báo lỗi mặc định, hoặc chuyển thành thiếu theo cấu hình |
| Dữ liệu mới | Dùng lại `preprocessor.joblib` đã fit, giữ nguyên schema đầu ra |

StandardScaler dùng mean/std train; RobustScaler dùng median/IQR train. MinMaxScaler không clip: giá trị mới vượt min/max train có thể nằm ngoài [0, 1]. Cột hằng không gây chia cho 0. Cột toàn thiếu trong train được giữ: cột số điền 0, cột phân loại dùng nhãn thiếu. Các trường hợp này được ghi trong manifest.

One-hot nhóm các nhãn ít gặp theo `max_categories`, giới hạn độ rộng. Giá trị chưa thấy trong train được mã hóa thành vector 0 của cột đó. Ordinal dùng -1 cho giá trị chưa thấy; mã số không khẳng định thứ tự nghiệp vụ. Giá trị phân loại thật được thêm tiền tố nội bộ để không trùng với nhãn dành cho dữ liệu thiếu.

Tỷ lệ test/validation được khai báo theo toàn bộ các dòng có target. Do làm tròn, tỷ lệ thực tế có thể hơi khác. Chia phân tầng cần đủ mẫu mỗi lớp ở mỗi lần chia; ứng dụng báo lỗi thay vì tự bỏ phân tầng. Chia thời gian không cho cùng timestamp nằm ở hai tập tại ranh giới chia.

## Cách dùng

Trong tab **6. Học máy**, chọn target, feature số/phân loại, phương pháp chuẩn hóa và tỷ lệ chia. Nhấn **Chuẩn bị dữ liệu ML**, kiểm tra số dòng/feature, phân phối train và ghi chú, tải ZIP. Đổi cấu hình sẽ vô hiệu hóa kết quả cũ.

CLI:

```powershell
.\.venv\Scripts\python.exe -m dataprep examples/sales_dirty.csv --ml-config examples/ml_config.json --output outputs/ml_demo
```

API:

```python
from dataprep.io import load_data
from dataprep.ml import MLConfig, prepare_ml, export_ml_bundle

data = load_data("examples/sales_dirty.csv")
cfg = MLConfig(
    target="quantity",
    numeric_columns=["age", "income"],
    categorical_columns=["city"],
    numeric_errors="coerce",
)
result = prepare_ml(data.frame, cfg)
X_train, y_train = result.matrices["train"], result.targets["train"]
X_test = result.matrices["test"]
# estimator.fit(X_train, y_train)
# predictions = estimator.predict(X_test)
```

## Đầu ra

- `X_train/validation/test.npz`: ma trận sparse CSR, cùng thứ tự feature giữa các tập.
- `X_*.csv`: thêm khi mỗi tập không quá 1 triệu ô để giới hạn RAM khi chuyển dense.
- `y_*.csv`: target; nhãn phân loại được mã số.
- `rows_*.csv`: vị trí dòng nguồn bắt đầu từ 0, cùng thứ tự với X/y.
- `preprocessor.joblib`: bộ parsing/imputer/scaler/encoder đã fit trên train.
- `target_encoder.joblib`: chỉ có khi phân loại.
- `ml_config.json`, `manifest.json`, `USAGE.md`: cấu hình, schema, mapping, phiên bản dependency, ghi chú.

Nạp joblib cần cùng phiên bản project/dependency; chỉ nạp artifact từ nguồn tin cậy. Ứng dụng không cung cấp chức năng upload/chạy joblib từ người khác.

## Giới hạn

Đây là dữ liệu cho đánh giá holdout. Với cross-validation/tuning, dùng sklearn Pipeline và fit lại preprocessing trong từng fold trên dữ liệu thô. Không dùng ma trận đã fit toàn train để giả định các fold validation độc lập.

Không tự chọn feature, cân bằng lớp/SMOTE, vector hóa văn bản, tạo feature ngày giờ, huấn luyện/tối ưu mô hình hoặc đảm bảo không có leakage theo nghiệp vụ. Bản ghi trùng giữa các tập được cảnh báo, không tự xóa. Chưa hỗ trợ GroupKFold/GroupShuffleSplit; dữ liệu lặp theo khách hàng, bệnh nhân hoặc thiết bị cần chiến lược chia nhóm riêng.

Tham khảo: https://scikit-learn.org/stable/common_pitfalls.html

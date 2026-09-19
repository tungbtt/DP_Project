"""Leakage-aware supervised learning preparation: split first, fit only on train."""

import json
import math
import platform
import zipfile
from dataclasses import asdict, dataclass, field
from io import BytesIO
from itertools import pairwise

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from . import __version__
from .errors import DataPrepError
from .serialization import dumps


@dataclass
class MLConfig:
    target: str
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    task: str = "regression"
    scaler: str = "standard"
    numeric_imputer: str = "median"
    encoding: str = "onehot"
    max_categories: int = 50
    test_size: float = 0.2
    validation_size: float = 0.2
    seed: int = 42
    split_method: str = "random"
    stratify: bool = True
    time_column: str = ""
    time_format: str = "ISO8601"
    drop_missing_target: bool = False
    numeric_errors: str = "raise"

    @classmethod
    def from_json(cls, text):
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError("Cấu hình cần là JSON object.")
            return cls(**payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataPrepError(f"Cấu hình ML không hợp lệ: {exc}") from exc


class TabularSanitizer(TransformerMixin, BaseEstimator):
    """Persisted deterministic parsing, also applied to new inference data."""

    def __init__(self, numeric_columns, categorical_columns, numeric_errors="raise"):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_errors = numeric_errors

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(self.numeric_columns + self.categorical_columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise DataPrepError("Preprocessor cần DataFrame có tên cột gốc.")
        columns = self.numeric_columns + self.categorical_columns
        missing = set(columns) - set(X.columns)
        if missing:
            raise DataPrepError("Thiếu feature đầu vào: " + ", ".join(sorted(missing)))
        if X.columns.duplicated().any():
            raise DataPrepError("Tên cột đầu vào ML không được trùng.")
        output = pd.DataFrame(index=X.index)
        for col in self.numeric_columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                raise DataPrepError(
                    f"'{col}' là ngày giờ; hãy loại khỏi feature số hoặc tạo feature thời gian riêng."
                )
            values = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            invalid = X[col].notna() & values.isna()
            if invalid.any() and self.numeric_errors == "raise":
                raise DataPrepError(
                    f"'{col}' có {int(invalid.sum())} giá trị không phải số hữu hạn. "
                    "Kiểm tra nguồn hoặc chọn chuyển giá trị lỗi thành thiếu."
                )
            output[col] = values.to_numpy(dtype=float, na_value=np.nan)
        for col in self.categorical_columns:
            # Prefix real values so a literal '__MISSING__' is distinct from missing.
            output[col] = X[col].map(lambda v: np.nan if pd.isna(v) else "value:" + str(v)).astype(object)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


@dataclass
class MLResult:
    matrices: dict
    targets: dict
    row_indices: dict
    feature_names: list[str]
    preprocessor: Pipeline
    target_encoder: LabelEncoder | None
    manifest: dict


def _validate_config(frame, config):
    if not isinstance(config.target, str) or config.target not in frame:
        raise DataPrepError("Cần chọn cột mục tiêu có trong dữ liệu.")
    for columns in (config.numeric_columns, config.categorical_columns):
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise DataPrepError("Danh sách feature phải gồm tên cột.")
    selected = config.numeric_columns + config.categorical_columns
    if not selected or len(set(selected)) != len(selected):
        raise DataPrepError("Cần ít nhất một feature và không được chọn trùng giữa cột số/phân loại.")
    if set(selected) - set(frame):
        raise DataPrepError("Có feature không tồn tại trong dữ liệu.")
    if config.target in selected:
        raise DataPrepError("Cột mục tiêu không được có trong feature.")
    enums = {
        "task": ("classification", "regression"),
        "scaler": ("standard", "minmax", "robust", "none"),
        "numeric_imputer": ("median", "mean", "zero"),
        "encoding": ("onehot", "ordinal"),
        "split_method": ("random", "time"),
        "numeric_errors": ("raise", "coerce"),
    }
    for key, allowed in enums.items():
        if getattr(config, key) not in allowed:
            raise DataPrepError(f"Giá trị {key} không hợp lệ.")
    for key in ("test_size", "validation_size"):
        value = getattr(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DataPrepError(f"{key} phải là số hữu hạn.")
    if not 0 < config.test_size < 0.5 or not 0 <= config.validation_size < 0.5:
        raise DataPrepError("Test cần >0 và <50%; validation từ 0 đến dưới 50%.")
    if config.test_size + config.validation_size >= 0.8:
        raise DataPrepError("Cần giữ trên 20% dữ liệu cho train.")
    if not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
        raise DataPrepError("Seed phải là số nguyên từ 0 đến 2^32-1.")
    if not isinstance(config.max_categories, int) or not 2 <= config.max_categories <= 200:
        raise DataPrepError("max_categories cần nằm trong khoảng 2–200.")
    for key in ("stratify", "drop_missing_target"):
        if not isinstance(getattr(config, key), bool):
            raise DataPrepError(f"{key} phải là true hoặc false.")
    if config.split_method == "time":
        if config.time_column not in frame or not config.time_format or config.time_format == "mixed":
            raise DataPrepError("Chia theo thời gian cần cột ngày và định dạng rõ ràng.")
        if config.time_column == config.target or config.time_column in selected:
            raise DataPrepError("Cột dùng để chia theo thời gian phải tách khỏi target và feature.")


def _split_rows(frame, y, config):
    rows = frame.index.to_numpy()
    if len(rows) < 5:
        raise DataPrepError("Cần ít nhất 5 dòng có nhãn để chia tập ML.")
    try:
        if config.split_method == "time":
            times = pd.to_datetime(
                frame[config.time_column], format=config.time_format, errors="raise", utc=True
            )
            if times.isna().any():
                raise DataPrepError("Cột chia thời gian có giá trị thiếu.")
            rows = times.sort_values(kind="stable").index.to_numpy()
            n_test = math.ceil(len(rows) * config.test_size)
            n_val = math.ceil(len(rows) * config.validation_size)
            n_train = len(rows) - n_test - n_val
            if n_train < 2:
                raise DataPrepError("Train cần ít nhất 2 dòng; giảm tỷ lệ validation/test.")
            train, val, test = rows[:n_train], rows[n_train : n_train + n_val], rows[n_train + n_val :]
            # Equal timestamps must not be split across partitions.
            partitions = [part for part in (train, val, test) if len(part)]
            for previous, following in pairwise(partitions):
                if times.loc[previous[-1]] >= times.loc[following[0]]:
                    raise DataPrepError(
                        "Ranh giới chia trùng timestamp. Đổi tỷ lệ hoặc tổng hợp theo thời điểm trước."
                    )
        else:
            stratified = config.task == "classification" and config.stratify
            train_val, test = train_test_split(
                rows,
                test_size=config.test_size,
                random_state=config.seed,
                stratify=y.loc[rows] if stratified else None,
            )
            if config.validation_size:
                train, val = train_test_split(
                    train_val,
                    test_size=config.validation_size / (1 - config.test_size),
                    random_state=config.seed,
                    stratify=y.loc[train_val] if stratified else None,
                )
            else:
                train, val = train_val, np.asarray([], dtype=int)
            if len(train) < 2:
                raise DataPrepError("Train cần ít nhất 2 dòng.")
        return {"train": train, "validation": val, "test": test}
    except ValueError as exc:
        if isinstance(exc, DataPrepError):
            raise
        raise DataPrepError(
            f"Không chia được tập dữ liệu: {exc}. "
            "Kiểm tra số mẫu mỗi lớp, tỷ lệ chia hoặc định dạng thời gian."
        ) from exc


def _make_preprocessor(config):
    transforms = []
    if config.numeric_columns:
        imputer = SimpleImputer(
            strategy="constant" if config.numeric_imputer == "zero" else config.numeric_imputer,
            fill_value=0,
            keep_empty_features=True,
        )
        scales = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": "passthrough",
        }
        transforms.append(
            (
                "numeric",
                Pipeline([("imputer", imputer), ("scaler", scales[config.scaler])]),
                config.numeric_columns,
            )
        )
    if config.categorical_columns:
        encoder = (
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=config.max_categories)
            if config.encoding == "onehot"
            else OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, max_categories=config.max_categories
            )
        )
        transforms.append(
            (
                "category",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__", keep_empty_features=True
                            ),
                        ),
                        ("encoder", encoder),
                    ]
                ),
                config.categorical_columns,
            )
        )
    columns = ColumnTransformer(transforms, remainder="drop", sparse_threshold=1.0)
    return Pipeline(
        [
            (
                "sanitize",
                TabularSanitizer(config.numeric_columns, config.categorical_columns, config.numeric_errors),
            ),
            ("columns", columns),
        ]
    )


def prepare_ml(original, config: MLConfig):
    """Caller supplies original loaded data, never globally fitted EDA results."""
    _validate_config(original, config)
    frame = original.copy(deep=True).reset_index(drop=True)
    missing_target = frame[config.target].isna()
    if missing_target.any() and not config.drop_missing_target:
        raise DataPrepError(
            f"Có {int(missing_target.sum())} dòng thiếu target; cần chọn loại những dòng này."
        )
    frame = frame.loc[~missing_target].copy()
    if config.task == "regression":
        y = pd.to_numeric(frame[config.target], errors="coerce").astype(float)
        if not np.isfinite(y.to_numpy()).all():
            raise DataPrepError("Target hồi quy cần là số hữu hạn; không tự điền hoặc sửa nhãn lỗi.")
    else:
        y = frame[config.target].astype(str)
        if y.nunique() < 2:
            raise DataPrepError("Phân loại cần ít nhất 2 lớp mục tiêu.")
    indices = _split_rows(frame, y, config)
    # Diagnose identical examples crossing partitions, without silently dropping real observations.
    selected = config.numeric_columns + config.categorical_columns
    signatures = pd.util.hash_pandas_object(frame[selected + [config.target]], index=False)
    sets = {name: set(signatures.loc[rows]) for name, rows in indices.items()}
    duplicate_overlap = len(
        (sets["train"] & sets["test"])
        | (sets["train"] & sets["validation"])
        | (sets["validation"] & sets["test"])
    )
    notes = ["Imputer, scaler và encoder chỉ fit trên train; validation/test chỉ transform."]
    if duplicate_overlap:
        notes.append(
            f"Có {duplicate_overlap} mẫu trùng feature + target giữa các tập. "
            "Kiểm tra bản ghi trùng hoặc quan hệ nhóm trước khi đánh giá mô hình."
        )
    preprocessor = _make_preprocessor(config)
    try:
        # No full-dataset fit or transform happens before partitioning.
        train_matrix = preprocessor.fit_transform(frame.loc[indices["train"], selected])
        feature_names = preprocessor.get_feature_names_out().tolist()
        matrices = {"train": sparse.csr_matrix(train_matrix)}
        for name in ("validation", "test"):
            matrices[name] = (
                sparse.csr_matrix(preprocessor.transform(frame.loc[indices[name], selected]))
                if len(indices[name])
                else sparse.csr_matrix((0, len(feature_names)))
            )
        if any(not np.isfinite(matrix.data).all() for matrix in matrices.values()):
            raise DataPrepError("Sau chuẩn hóa có số không hữu hạn; kiểm tra độ lớn các giá trị nguồn.")
        target_encoder = LabelEncoder() if config.task == "classification" else None
        if target_encoder is not None:
            target_encoder.fit(y.loc[indices["train"]])
            if len(target_encoder.classes_) < 2:
                raise DataPrepError("Train chỉ có một lớp. Đổi cách chia tập hoặc bổ sung dữ liệu.")
            unknown = set(y) - set(target_encoder.classes_)
            if unknown:
                raise DataPrepError(
                    "Validation/test có lớp chưa xuất hiện ở train. Dùng chia phân tầng hoặc bổ sung dữ liệu."
                )
        targets = {
            name: pd.Series(
                target_encoder.transform(y.loc[rows])
                if target_encoder is not None and len(rows)
                else y.loc[rows].to_numpy(),
                name="target",
            )
            for name, rows in indices.items()
        }
    except (ValueError, TypeError) as exc:
        if isinstance(exc, DataPrepError):
            raise
        raise DataPrepError(f"Không chuẩn bị được dữ liệu ML: {exc}") from exc
    clean_train = preprocessor.named_steps["sanitize"].transform(frame.loc[indices["train"], selected])
    empty_train = [c for c in selected if clean_train[c].isna().all()]
    if empty_train:
        notes.append(
            "Cột toàn thiếu trong train: "
            + ", ".join(empty_train)
            + ". Cột số điền 0; cột phân loại dùng nhãn thiếu cố định."
        )
    invalid_counts = {}
    for col in config.numeric_columns:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        invalid_counts[col] = int((frame[col].notna() & values.isna()).sum())
    if config.encoding == "ordinal":
        notes.append(
            "Ordinal tạo mã số, không suy ra thứ tự nghiệp vụ; giá trị chưa thấy ở train được mã -1."
        )
    else:
        notes.append(
            "One-hot: giá trị chưa thấy ở train trở thành vector 0 cho cột đó; nhãn ít gặp được gộp theo giới hạn đã chọn."
        )
    if config.scaler == "minmax":
        notes.append("MinMax dùng min/max train; giá trị ngoài miền train có thể ra ngoài [0, 1].")
    manifest = {
        "version": 1,
        "config": asdict(config),
        "source_rows": len(original),
        "excluded_target_rows": np.flatnonzero(missing_target.to_numpy()).tolist(),
        "split_rows": {k: len(v) for k, v in indices.items()},
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "fit_partition": "train",
        "source": "original_loaded_data",
        "target_mapping": {str(label): int(i) for i, label in enumerate(target_encoder.classes_)}
        if target_encoder is not None
        else None,
        "invalid_numeric_to_missing": invalid_counts,
        "all_missing_train_columns": empty_train,
        "duplicate_signatures_across_splits": duplicate_overlap,
        "notes": notes,
        "versions": {
            "python": platform.python_version(),
            "data-prep-studio": __version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    return MLResult(matrices, targets, indices, feature_names, preprocessor, target_encoder, manifest)


def export_ml_bundle(result, max_csv_cells=1_000_000):
    """Always export sparse matrices; dense CSV is optional and bounded."""
    stream = BytesIO()
    csv_splits = []
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, matrix in result.matrices.items():
            binary = BytesIO()
            sparse.save_npz(binary, matrix)
            archive.writestr(f"X_{name}.npz", binary.getvalue())
            archive.writestr(f"y_{name}.csv", result.targets[name].to_csv(index=False).encode("utf-8-sig"))
            archive.writestr(
                f"rows_{name}.csv",
                pd.Series(result.row_indices[name], name="source_row_position")
                .to_csv(index=False)
                .encode("utf-8-sig"),
            )
            if matrix.shape[0] * matrix.shape[1] <= max_csv_cells:
                archive.writestr(
                    f"X_{name}.csv",
                    pd.DataFrame(matrix.toarray(), columns=result.feature_names)
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                )
                csv_splits.append(name)
        fitted = BytesIO()
        joblib.dump(result.preprocessor, fitted, compress=3)
        archive.writestr("preprocessor.joblib", fitted.getvalue())
        if result.target_encoder is not None:
            encoded = BytesIO()
            joblib.dump(result.target_encoder, encoded)
            archive.writestr("target_encoder.joblib", encoded.getvalue())
        archive.writestr("ml_config.json", dumps(result.manifest["config"]))
        archive.writestr(
            "manifest.json",
            dumps(
                {**result.manifest, "dense_csv_splits": csv_splits, "max_csv_cells_per_split": max_csv_cells}
            ),
        )
        archive.writestr(
            "USAGE.md",
            """# Dữ liệu sẵn sàng cho học máy

Các hàng X/y/rows cùng tập có cùng thứ tự. rows_* là vị trí dòng nguồn bắt đầu từ 0 sau khi đọc file.
Train/validation/test đã được chia trước khi fit imputer, scaler và encoder. Target phân loại đã mã hóa;
mapping nằm trong manifest.json. File NPZ lưu ma trận sparse; CSV chỉ xuất nếu không vượt giới hạn ô.

```python
from scipy.sparse import load_npz
import pandas as pd
import joblib
X_train = load_npz('X_train.npz')
y_train = pd.read_csv('y_train.csv')['target']
X_test = load_npz('X_test.npz')
y_test = pd.read_csv('y_test.csv')['target']
# Cài cùng phiên bản project và dependency trong manifest trước khi nạp transformer.
# Chỉ nạp joblib từ nguồn bạn tin cậy.
preprocessor = joblib.load('preprocessor.joblib')
# new_data cần DataFrame có tên các feature gốc; không cần target.
# X_new = preprocessor.transform(new_data)
```

Các ma trận phù hợp cho đánh giá holdout. Khi cross-validation hoặc tuning, clone preprocessor và fit lại
trong từng fold trên dữ liệu thô; không dùng thống kê đã fit trên toàn train cho các fold validation.
Không tự động giải quyết rò rỉ do cùng khách hàng/nhóm, bản ghi trùng hoặc feature tiết lộ nhãn.
""",
        )
    return stream.getvalue()

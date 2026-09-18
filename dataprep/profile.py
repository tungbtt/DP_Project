"""Descriptive statistics and transparent, rule-based recommendations."""

import numpy as np
import pandas as pd

from .errors import DataPrepError
from .io import is_id_name
from .serialization import json_safe

ROLES = ("numeric", "category", "text", "datetime", "id", "ignore")


def infer_role(series):
    if is_id_name(str(series.name)):
        return "id"
    if pd.api.types.is_bool_dtype(series):
        return "category"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    count = series.nunique(dropna=True)
    return "category" if count <= 30 or count / max(len(series), 1) < 0.05 else "text"


def resolve_roles(frame, roles=None):
    roles = roles or {}
    if set(roles) - set(frame.columns):
        raise DataPrepError("Vai trò tham chiếu cột không tồn tại: " + ", ".join(set(roles) - set(frame)))
    if any(v not in ROLES for v in roles.values()):
        raise DataPrepError("Vai trò cột không hợp lệ.")
    return {c: roles.get(c, infer_role(frame[c])) for c in frame}


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def iqr_bounds(series, factor=1.5):
    values = numeric_values(series).dropna()
    if len(values) < 4:
        return None
    q1, q3 = values.quantile([0.25, 0.75])
    if q3 == q1:
        return None
    return float(q1 - factor * (q3 - q1)), float(q3 + factor * (q3 - q1))


def profile_data(frame, roles=None):
    roles = resolve_roles(frame, roles)
    n, p = frame.shape
    missing = int(frame.isna().sum().sum())
    result = {
        "overview": {
            "rows": n,
            "columns": p,
            "cells": n * p,
            "missing_cells": missing,
            "missing_pct": 100 * missing / max(n * p, 1),
            "duplicate_rows": int(frame.duplicated().sum()),
            "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        },
        "columns": [],
        "issues": [],
        "roles": roles,
    }

    def issue(col, kind, count, message, suggestion):
        result["issues"].append(
            {"column": col, "kind": kind, "count": int(count), "message": message, "suggestion": suggestion}
        )

    if result["overview"]["duplicate_rows"]:
        issue(
            "(toàn bảng)",
            "duplicates",
            result["overview"]["duplicate_rows"],
            "Bản ghi lặp lại toàn bộ giá trị (đếm các bản sao dư).",
            "Kiểm tra ý nghĩa bản ghi trước khi loại trùng.",
        )
    for col in frame:
        s = frame[col]
        role = roles[col]
        entry = {
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "valid": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100) if n else 0,
            "unique": int(s.nunique()),
            "stats": {},
        }
        if entry["missing"]:
            issue(
                col,
                "missing",
                entry["missing"],
                "Có giá trị thiếu.",
                "Cân nhắc điền median/mean theo phân phối."
                if role == "numeric"
                else "Giữ thiếu hoặc chọn nhãn thay thế phù hợp nghiệp vụ.",
            )
        if entry["unique"] == 1:
            issue(
                col,
                "constant",
                entry["valid"],
                "Cột chỉ có một giá trị khác thiếu.",
                "Kiểm tra vai trò trước khi loại cột.",
            )
        if role == "numeric":
            vals = numeric_values(s)
            invalid = s.notna() & vals.isna()
            if invalid.any():
                issue(
                    col,
                    "invalid_numeric",
                    invalid.sum(),
                    "Không phải số hữu hạn.",
                    "Chọn chuyển kiểu và chính sách xử lý giá trị lỗi.",
                )
            entry["stats"] = vals.describe().to_dict()
            entry["stats"]["median"] = vals.median() if vals.notna().any() else None
            bounds = iqr_bounds(vals)
            if bounds:
                mask = vals.lt(bounds[0]) | vals.gt(bounds[1])
                entry["stats"]["iqr_bounds"] = list(bounds)
                entry["stats"]["outliers"] = int(mask.sum())
                if mask.any():
                    issue(
                        col,
                        "outliers",
                        mask.sum(),
                        "Nằm ngoài ngưỡng IQR × 1.5.",
                        "Đây là ngoại lệ thống kê; mặc định giữ nguyên, không tự xóa.",
                    )
        elif role == "datetime":
            if pd.api.types.is_datetime64_any_dtype(s):
                entry["stats"] = {"min": s.min(), "max": s.max()}
            else:
                issue(
                    col,
                    "datetime_unparsed",
                    entry["valid"],
                    "Cột thời gian chưa được chuyển kiểu.",
                    "Chọn định dạng ngày rõ ràng trong bước chuyển kiểu.",
                )
        elif role != "ignore":
            strings = s.astype("string")
            counts = strings.value_counts().head(10)
            entry["stats"] = {
                "top": [{"value": str(k), "count": int(v)} for k, v in counts.items()],
                "mean_length": strings.str.len().mean(),
            }
            whitespace = strings.ne(strings.str.strip()).fillna(False)
            if whitespace.any():
                issue(
                    col,
                    "whitespace",
                    whitespace.sum(),
                    "Có khoảng trắng đầu/cuối.",
                    "Dùng chuẩn hóa chuỗi: trim.",
                )
            numeric = pd.to_numeric(strings.str.strip(), errors="coerce")
            ratio = numeric.notna().sum() / max(entry["valid"], 1)
            if role != "id" and 0.8 <= ratio < 1:
                issue(
                    col,
                    "mixed_numeric",
                    entry["valid"] - numeric.notna().sum(),
                    "Phần lớn giá trị giống số nhưng có giá trị không chuyển được.",
                    "Kiểm tra các giá trị lỗi trước khi chuyển sang số.",
                )
        result["columns"].append(entry)
    return json_safe(result)


def correlation(frame, roles=None, method="pearson"):
    if method not in ("pearson", "spearman"):
        raise DataPrepError("Chỉ hỗ trợ Pearson hoặc Spearman.")
    resolved = resolve_roles(frame, roles)
    cols = [c for c in frame if resolved[c] == "numeric"]
    numeric = pd.DataFrame({c: numeric_values(frame[c]) for c in cols})
    return numeric.corr(method=method, min_periods=3)


def compare_profiles(before, after):
    labels = {
        "rows": "Số dòng",
        "columns": "Số cột",
        "cells": "Số ô",
        "missing_cells": "Ô thiếu",
        "missing_pct": "Tỷ lệ thiếu (%)",
        "duplicate_rows": "Bản sao dư",
    }
    return pd.DataFrame(
        [
            {
                "Chỉ số": label,
                "Trước": before["overview"][key],
                "Sau": after["overview"][key],
                "Chênh lệch": after["overview"][key] - before["overview"][key],
            }
            for key, label in labels.items()
        ]
    )

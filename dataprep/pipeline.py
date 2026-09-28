"""Deterministic transformations on copies; failed runs return no partial result."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DataPrepError
from .profile import iqr_bounds, numeric_values, resolve_roles
from .serialization import json_safe

OPERATIONS = {
    "normalize_text": {"columns", "case", "empty_as_missing"},
    "replace_values": {"columns", "mapping"},
    "cast": {"columns", "dtype", "errors", "format", "decimal", "thousands"},
    "fill_missing": {"columns", "strategy", "value"},
    "drop_missing": {"columns", "how"},
    "drop_duplicates": {"columns", "keep"},
    "drop_columns": {"columns"},
    "outliers": {"columns", "method", "factor"},
    "range": {"columns", "min", "max", "action"},
}


@dataclass
class PipelineResult:
    frame: pd.DataFrame
    log: list[dict]
    config: dict


def validate_config(config):
    if not isinstance(config, dict):
        raise DataPrepError("Pipeline phải là JSON object.")
    if set(config) - {"version", "roles", "steps", "expected_columns"}:
        raise DataPrepError("Pipeline có thuộc tính không được hỗ trợ.")
    if config.get("version", 1) != 1:
        raise DataPrepError("Chỉ hỗ trợ pipeline version 1.")
    steps = config.get("steps", [])
    if not isinstance(steps, list) or len(steps) > 100:
        raise DataPrepError("steps phải là danh sách, tối đa 100 bước.")
    roles = config.get("roles", {})
    if not isinstance(roles, dict):
        raise DataPrepError("roles phải là object ánh xạ tên cột → vai trò.")
    expected = config.get("expected_columns", [])
    if not isinstance(expected, list) or not all(isinstance(c, str) for c in expected):
        raise DataPrepError("expected_columns phải là danh sách tên cột.")
    for index, step in enumerate(steps, 1):
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("op"), str)
            or step.get("op") not in OPERATIONS
        ):
            raise DataPrepError(f"Bước {index}: phép biến đổi không được hỗ trợ.")
        if set(step) - (OPERATIONS[step["op"]] | {"op"}):
            raise DataPrepError(f"Bước {index}: tham số không được hỗ trợ.")
        cols = step.get("columns", [])
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise DataPrepError(f"Bước {index}: columns phải là danh sách tên cột.")
        if len(set(cols)) != len(cols):
            raise DataPrepError(f"Bước {index}: columns có tên trùng.")
        if not cols and step["op"] not in ("drop_duplicates", "drop_missing"):
            raise DataPrepError(f"Bước {index}: cần chọn ít nhất một cột.")
    return config


def parse_config(text):
    try:
        return validate_config(json.loads(text))
    except json.JSONDecodeError as exc:
        raise DataPrepError(f"Pipeline JSON lỗi dòng {exc.lineno}: {exc.msg}") from exc


def _number_series(series):
    values = numeric_values(series)
    if (series.notna() & values.isna()).any():
        raise DataPrepError(f"Cột '{series.name}' chứa giá trị không phải số hữu hạn; chuyển kiểu trước.")
    return values


def _apply(frame, step):
    op = step["op"]
    cols = step.get("columns") or list(frame.columns)
    absent = set(cols) - set(frame.columns)
    if absent:
        raise DataPrepError("Cột không tồn tại: " + ", ".join(sorted(absent)))
    if op == "drop_columns":
        if len(cols) == len(frame.columns):
            raise DataPrepError("Không được xóa toàn bộ cột.")
        return frame.drop(columns=cols)
    if op == "drop_duplicates":
        keep = step.get("keep", "first")
        if keep not in ("first", "last", False):
            raise DataPrepError("keep phải là first, last hoặc false.")
        return frame.drop_duplicates(subset=cols, keep=keep)
    if op == "drop_missing":
        how = step.get("how", "any")
        if how not in ("any", "all"):
            raise DataPrepError("how phải là any hoặc all.")
        return frame.dropna(subset=cols, how=how)
    for col in cols:
        s = frame[col]
        if op == "normalize_text":
            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                raise DataPrepError(f"'{col}' không phải cột chuỗi.")
            if not s.dropna().map(lambda v: isinstance(v, str)).all():
                raise DataPrepError(f"'{col}' chứa giá trị không phải chuỗi.")
            values = s.astype("string").str.strip()
            case = step.get("case", "keep")
            if case not in ("keep", "lower", "upper", "casefold"):
                raise DataPrepError("case không hợp lệ.")
            if case != "keep":
                values = getattr(values.str, case)()
            if step.get("empty_as_missing", True):
                values = values.replace("", pd.NA)
            frame[col] = values
        elif op == "replace_values":
            mapping = step.get("mapping")
            if not isinstance(mapping, dict):
                raise DataPrepError("mapping phải là JSON object.")
            frame[col] = s.map(lambda v, mapping=mapping: mapping.get(str(v), v) if pd.notna(v) else v)
        elif op == "cast":
            dtype = step.get("dtype", "numeric")
            errors = step.get("errors", "raise")
            if errors not in ("raise", "coerce"):
                raise DataPrepError("errors phải là raise hoặc coerce.")
            if dtype == "numeric":
                normalized = s.astype("string").str.strip()
                decimal, thousands = step.get("decimal", "."), step.get("thousands", "")
                if decimal not in (".", ",") or (thousands and (len(thousands) != 1 or thousands == decimal)):
                    raise DataPrepError("Dấu thập phân/hàng nghìn không hợp lệ.")
                if thousands:
                    normalized = normalized.str.replace(thousands, "", regex=False)
                if decimal != ".":
                    normalized = normalized.str.replace(decimal, ".", regex=False)
                values = pd.to_numeric(normalized, errors="coerce").replace([np.inf, -np.inf], np.nan)
            elif dtype == "datetime":
                fmt = step.get("format")
                if not fmt or fmt == "mixed":
                    raise DataPrepError("Cần định dạng ngày rõ ràng, ví dụ %d/%m/%Y hoặc ISO8601.")
                values = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
            elif dtype == "string":
                values = s.astype("string")
            else:
                raise DataPrepError("dtype phải là numeric, datetime hoặc string.")
            invalid = s.notna() & values.isna()
            if errors == "raise" and invalid.any():
                examples = s[invalid].astype(str).head(3).tolist()
                raise DataPrepError(
                    f"'{col}': {invalid.sum()} giá trị chuyển kiểu thất bại, ví dụ {examples}."
                )
            frame[col] = values
        elif op == "fill_missing":
            strategy = step.get("strategy", "median")
            if strategy not in ("median", "mean", "mode", "constant"):
                raise DataPrepError("Chiến lược điền thiếu không hợp lệ.")
            if not s.isna().any():
                continue
            if strategy in ("median", "mean"):
                values = _number_series(s)
                if values.dropna().empty:
                    raise DataPrepError(f"'{col}' toàn giá trị thiếu; cần điền hằng số.")
                fill = getattr(values, strategy)()
                frame[col] = values.astype(float).fillna(fill)
            elif strategy == "mode":
                mode = s.mode(dropna=True)
                if mode.empty:
                    raise DataPrepError(f"'{col}' toàn giá trị thiếu; không xác định được mode.")
                frame[col] = s.fillna(mode.iloc[0])
            else:
                fill = step.get("value")
                if fill is None or isinstance(fill, (list, dict)):
                    raise DataPrepError("Giá trị điền phải là một hằng số khác null.")
                if pd.api.types.is_numeric_dtype(s):
                    fill = float(fill)
                    if not np.isfinite(fill):
                        raise DataPrepError("Giá trị điền phải hữu hạn.")
                    frame[col] = s.astype(float).fillna(fill)
                elif pd.api.types.is_datetime64_any_dtype(s):
                    raise DataPrepError("Điền thời gian bằng mode hoặc chuyển sang chuỗi trước.")
                else:
                    frame[col] = s.astype("string").fillna(str(fill))
        elif op == "outliers":
            values = _number_series(s)
            factor = float(step.get("factor", 1.5))
            method = step.get("method", "clip")
            if not np.isfinite(factor) or factor <= 0 or method not in ("clip", "drop", "missing"):
                raise DataPrepError("IQR cần factor > 0 và method clip/drop/missing.")
            bounds = iqr_bounds(values, factor)
            if bounds is None:
                continue
            mask = values.lt(bounds[0]) | values.gt(bounds[1])
            if method == "drop":
                frame = frame.loc[~mask].copy()
            elif method == "missing":
                frame[col] = values.mask(mask)
            else:
                frame[col] = values.astype(float).clip(*bounds)
        elif op == "range":
            values = _number_series(s)
            low, high = step.get("min"), step.get("max")
            if low is None and high is None:
                raise DataPrepError("Cần khai báo min hoặc max.")
            if low is not None:
                low = float(low)
            if high is not None:
                high = float(high)
            if any(v is not None and not np.isfinite(v) for v in (low, high)):
                raise DataPrepError("Ngưỡng phải hữu hạn.")
            if low is not None and high is not None and low > high:
                raise DataPrepError("min không được lớn hơn max.")
            mask = pd.Series(False, index=frame.index)
            if low is not None:
                mask |= values.lt(low)
            if high is not None:
                mask |= values.gt(high)
            action = step.get("action", "missing")
            if action == "missing":
                frame[col] = values.mask(mask)
            elif action == "drop":
                frame = frame.loc[~mask].copy()
            else:
                raise DataPrepError("action phải là missing hoặc drop.")
    return frame


def changed_cells(before, after):
    cols = before.columns.intersection(after.columns)
    rows = before.index.intersection(after.index)
    changed = 0
    # Bound temporary strings to one column and one chunk, even for millions of rows.
    for col in cols:
        for start in range(0, len(rows), 50_000):
            selected = rows[start : start + 50_000]
            a, b = before.loc[selected, col], after.loc[selected, col]
            equal = a.astype("string").eq(b.astype("string")).fillna(False) | (a.isna() & b.isna())
            changed += int((~equal).sum())
    return changed


def run_pipeline(original, config):
    config = validate_config(config)
    expected = config.get("expected_columns", [])
    if expected and list(original.columns) != expected:
        raise DataPrepError("Cấu trúc cột không khớp expected_columns của pipeline.")
    resolve_roles(original, config.get("roles"))
    frame = original.copy(deep=True).reset_index(drop=True)
    log = []
    for index, step in enumerate(config.get("steps", []), 1):
        # _apply replaces columns/frames; it never mutates shared column buffers in place.
        before = frame.copy(deep=False)
        try:
            frame = _apply(frame, step)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise DataPrepError(f"Bước {index} ({step['op']}): {exc}") from exc
        log.append(
            {
                "step": index,
                "operation": step["op"],
                "parameters": step,
                "rows_before": len(before),
                "rows_after": len(frame),
                "removed_rows": len(before) - len(frame),
                "removed_columns": list(before.columns.difference(frame.columns)),
                "changed_cells_retained": changed_cells(before, frame),
                "missing_before": sum(int(before[c].isna().sum()) for c in before),
                "missing_after": sum(int(frame[c].isna().sum()) for c in frame),
                "dtype_changes": {
                    c: [str(before[c].dtype), str(frame[c].dtype)]
                    for c in frame
                    if str(before[c].dtype) != str(frame[c].dtype)
                },
            }
        )
    return PipelineResult(frame, json_safe(log), json_safe(config))

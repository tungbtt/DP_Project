"""Bounded tabular ingestion. Never silently skip malformed records."""

import csv
import json
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from .errors import DataPrepError

MAX_BYTES = 100 * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 200


@dataclass
class LoadOptions:
    encoding: str = "utf-8-sig"
    delimiter: str | None = None
    decimal: str = "."
    missing_tokens: list[str] = field(default_factory=list)
    json_path: str = ""
    sheet: str | int = 0
    table: str = ""
    infer_numeric: bool = True


@dataclass
class Dataset:
    frame: pd.DataFrame
    name: str
    metadata: dict


def is_id_name(name: str) -> bool:
    name = name.lower().strip()
    return bool(re.search(r"(^id$|_id$|^id_|^mã|^ma_|code$|phone|postal|zip)", name))


def _validate(frame):
    if len(frame) > MAX_ROWS:
        raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng. Hãy chia nhỏ dữ liệu.")
    if len(frame.columns) > MAX_COLUMNS:
        raise DataPrepError(f"Vượt giới hạn {MAX_COLUMNS} cột.")
    if not len(frame.columns) or frame.empty:
        raise DataPrepError("Dữ liệu rỗng hoặc không có bản ghi.")
    frame.columns = [str(c) for c in frame.columns]
    if frame.columns.duplicated().any() or any(not c.strip() for c in frame.columns):
        raise DataPrepError("Tên cột phải có nội dung và không được trùng nhau.")


def _infer(frame, decimal):
    """Only infer fully parseable numbers; preserve IDs and leading zero strings."""
    for col in frame:
        series = frame[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str)
        if values.empty or is_id_name(col) or values.str.strip().str.match(r"^[+-]?0\d").any():
            continue
        normalized = series.astype("string").str.strip()
        if decimal == ",":
            normalized = normalized.str.replace(",", ".", regex=False)
        parsed = pd.to_numeric(normalized, errors="coerce")
        if parsed.notna().sum() == series.notna().sum():
            frame[col] = parsed
    return frame


def _json_records(payload, options, lines=False):
    raw = payload.decode(options.encoding)
    if lines:
        data = []
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataPrepError(f"JSONL lỗi tại dòng {number}: {exc.msg}") from exc
            if len(data) > MAX_ROWS:
                raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")
    else:
        data = json.loads(raw)
        for part in filter(None, options.json_path.split(".")):
            if not isinstance(data, dict) or part not in data:
                raise DataPrepError(f"Không tìm thấy nhánh JSON: {options.json_path}")
            data = data[part]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise DataPrepError("JSON cần là danh sách object; hãy chọn nhánh chứa các bản ghi.")
    if len(data) > MAX_ROWS:
        raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")

    # Reject collisions such as {"a.b": 1, "a": {"b": 2}} instead of losing a value.
    def flat_record(record):
        output = {}

        def visit(obj, prefix=""):
            for key, value in obj.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict) and value:
                    visit(value, name)
                else:
                    if name in output:
                        raise DataPrepError(f"Tên cột JSON bị trùng sau khi làm phẳng: {name}")
                    output[name] = value

        visit(record)
        return output

    return pd.DataFrame([flat_record(row) for row in data])


def _sqlite(payload, table="", list_only=False):
    # Materialize only the supplied database, open read-only, never execute SQL uploads.
    with tempfile.TemporaryDirectory(prefix="dataprep-") as folder:
        path = Path(folder) / "source.sqlite"
        path.write_bytes(payload)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            if list_only:
                return names
            if not names:
                raise DataPrepError("SQLite không có bảng dữ liệu.")
            if not table and len(names) != 1:
                raise DataPrepError("Cần chọn bảng SQLite: " + ", ".join(names))
            table = table or names[0]
            if table not in names:
                raise DataPrepError(f"Không tìm thấy bảng: {table}")
            quoted = table.replace('"', '""')
            return pd.read_sql_query(f'SELECT * FROM "{quoted}" LIMIT {MAX_ROWS + 1}', conn)


def sqlite_tables(payload):
    if len(payload) > MAX_BYTES:
        raise DataPrepError("File vượt giới hạn 100 MiB.")
    try:
        return _sqlite(payload, list_only=True)
    except sqlite3.Error as exc:
        raise DataPrepError(f"Không đọc được SQLite: {exc}") from exc


def load_data(
    source: str | Path | bytes, filename: str | None = None, options: LoadOptions | None = None
) -> Dataset:
    options = options or LoadOptions()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_BYTES:
            raise DataPrepError("File vượt giới hạn 100 MiB.")
        payload, filename = path.read_bytes(), filename or path.name
    else:
        payload = source
    filename = filename or "data.csv"
    if not payload:
        raise DataPrepError("File rỗng.")
    if len(payload) > MAX_BYTES:
        raise DataPrepError("File vượt giới hạn 100 MiB.")
    ext = Path(filename).suffix.lower()
    if options.decimal not in (".", ","):
        raise DataPrepError("Dấu thập phân phải là '.' hoặc ','.")
    try:
        if ext in (".csv", ".tsv"):
            raw = payload.decode(options.encoding)
            delimiter = options.delimiter or ("\t" if ext == ".tsv" else None)
            if delimiter is None:
                try:
                    delimiter = csv.Sniffer().sniff(raw[:65536], delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
            if len(delimiter) != 1:
                raise DataPrepError("Dấu phân cách cần là một ký tự.")
            rows = csv.reader(StringIO(raw), delimiter=delimiter, strict=True)
            header = next(rows, [])
            if not header or len(set(header)) != len(header) or any(not c.strip() for c in header):
                raise DataPrepError("CSV cần tiêu đề không rỗng, không trùng tên.")
            count = 0
            for row in rows:
                if not row:
                    continue
                count += 1
                if len(row) != len(header):
                    raise DataPrepError(f"Dòng CSV {rows.line_num}: số trường khác tiêu đề.")
                if count > MAX_ROWS:
                    raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")
            frame = pd.read_csv(
                StringIO(raw),
                sep=delimiter,
                dtype="string",
                keep_default_na=False,
                na_values=["", *options.missing_tokens],
                on_bad_lines="error",
            )
        elif ext in (".json", ".jsonl", ".ndjson"):
            frame = _json_records(payload, options, ext != ".json")
        elif ext == ".xlsx":
            frame = pd.read_excel(
                BytesIO(payload),
                sheet_name=options.sheet,
                engine="openpyxl",
                dtype=object,
                keep_default_na=False,
                na_values=["", *options.missing_tokens],
                nrows=MAX_ROWS + 1,
            )
        elif ext in (".db", ".sqlite", ".sqlite3"):
            frame = _sqlite(payload, options.table)
        elif ext == ".parquet":
            frame = pd.read_parquet(BytesIO(payload))
        else:
            raise DataPrepError(
                "Định dạng chưa hỗ trợ. Dùng CSV, TSV, JSON/JSONL, XLSX, SQLite hoặc Parquet."
            )
        _validate(frame)
        # Nested arrays remain a single JSON value; no implicit explode or row multiplication.
        for col in frame:
            if frame[col].dtype == object:
                frame[col] = frame[col].map(
                    lambda v: (
                        json.dumps(v, ensure_ascii=False, sort_keys=True)
                        if isinstance(v, (dict, list))
                        else v
                    )
                )
        frame = frame.replace({"": pd.NA, **{v: pd.NA for v in options.missing_tokens}})
        if options.infer_numeric:
            frame = _infer(frame, options.decimal)
        frame = frame.reset_index(drop=True)
        return Dataset(
            frame,
            filename,
            {
                "source": filename,
                "bytes": len(payload),
                "format": ext.lstrip("."),
                "rows": len(frame),
                "columns": len(frame.columns),
                "load_options": vars(options),
                "sampled": False,
            },
        )
    except DataPrepError:
        raise
    except (ValueError, UnicodeError, csv.Error, sqlite3.Error, ImportError, OSError, BadZipFile) as exc:
        raise DataPrepError(f"Không đọc được dữ liệu: {exc}") from exc

"""Bounded tabular ingestion. Never silently skip malformed records."""

import csv
import json
import re
import sqlite3
import tempfile
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from io import BytesIO, TextIOWrapper
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from .errors import DataPrepError

MAX_BYTES = 512 * 1024 * 1024
MAX_ROWS = 2_000_000
MAX_COLUMNS = 200
CHUNK_ROWS = 50_000


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
    return bool(re.search(r"(^id$|_id$|^id_|^sbd$|^mã|^ma_|code$|phone|postal|zip)", name))


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


def _json_records(stream, options, lines=False):
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

    text = TextIOWrapper(stream, encoding=options.encoding)
    try:
        if not lines:
            data = json.load(text)
            for part in filter(None, options.json_path.split(".")):
                if not isinstance(data, dict) or part not in data:
                    raise DataPrepError(f"Không tìm thấy nhánh JSON: {options.json_path}")
                data = data[part]
            if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
                raise DataPrepError("JSON cần là danh sách object; hãy chọn nhánh chứa các bản ghi.")
            if len(data) > MAX_ROWS:
                raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")
            return pd.DataFrame.from_records(flat_record(row) for row in data)
        chunks, batch, count = [], [], 0
        for number, line in enumerate(text, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataPrepError(f"JSONL lỗi tại dòng {number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise DataPrepError(f"JSONL dòng {number} phải là object.")
            count += 1
            if count > MAX_ROWS:
                raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")
            batch.append(flat_record(row))
            if len(batch) == CHUNK_ROWS:
                chunks.append(pd.DataFrame.from_records(batch))
                batch = []
        if batch:
            chunks.append(pd.DataFrame.from_records(batch))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    finally:
        text.detach()


def _sqlite(source, table="", list_only=False):
    # Materialize only the supplied database, open read-only, never execute SQL uploads.
    is_path = isinstance(source, (str, Path))
    with nullcontext(None) if is_path else tempfile.TemporaryDirectory(prefix="dataprep-") as folder:
        path = Path(source).resolve() if is_path else Path(folder) / "source.sqlite"
        if not is_path:
            path.write_bytes(source)
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
    try:
        size = Path(payload).stat().st_size if isinstance(payload, (str, Path)) else len(payload)
        if size > MAX_BYTES:
            raise DataPrepError(f"File vượt giới hạn {MAX_BYTES // (1024 * 1024)} MiB.")
        return _sqlite(payload, list_only=True)
    except (sqlite3.Error, OSError) as exc:
        raise DataPrepError(f"Không đọc được SQLite: {exc}") from exc


def load_data(
    source: str | Path | bytes, filename: str | None = None, options: LoadOptions | None = None
) -> Dataset:
    options = options or LoadOptions()
    if isinstance(source, (str, Path)):
        path = Path(source)
        size, filename = path.stat().st_size, filename or path.name
    else:
        size = len(source)
    filename = filename or "data.csv"
    if not size:
        raise DataPrepError("File rỗng.")
    if size > MAX_BYTES:
        raise DataPrepError(f"File vượt giới hạn {MAX_BYTES // (1024 * 1024)} MiB.")
    ext = Path(filename).suffix.lower()
    if options.decimal not in (".", ","):
        raise DataPrepError("Dấu thập phân phải là '.' hoặc ','.")
    try:
        # Read paths directly: do not retain a file-sized bytes + decoded string copy.
        with open(source, "rb") if isinstance(source, (str, Path)) else BytesIO(source) as stream:
            frame = _read_frame(stream, source, ext, options)
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
                "bytes": size,
                "format": ext.lstrip("."),
                "rows": len(frame),
                "columns": len(frame.columns),
                "load_options": vars(options),
                "sampled": False,
                "memory_bytes": int(frame.memory_usage(deep=True).sum()),
            },
        )
    except DataPrepError:
        raise
    except MemoryError as exc:
        raise DataPrepError(
            "Không đủ RAM để nạp bảng. Dùng máy có RAM lớn hơn hoặc giảm số cột/kích thước file."
        ) from exc
    except (ValueError, UnicodeError, csv.Error, sqlite3.Error, ImportError, OSError, BadZipFile) as exc:
        raise DataPrepError(f"Không đọc được dữ liệu: {exc}") from exc


def _read_frame(stream, source, ext, options):
    if ext in (".csv", ".tsv"):
        text = TextIOWrapper(stream, encoding=options.encoding, newline="")
        try:
            sample = text.read(65536)
            text.seek(0)
            delimiter = options.delimiter or ("\t" if ext == ".tsv" else None)
            if delimiter is None:
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
            if len(delimiter) != 1:
                raise DataPrepError("Dấu phân cách cần là một ký tự.")
            rows = csv.reader(text, delimiter=delimiter, strict=True)
            header = next(rows, [])
            if not header or len(set(header)) != len(header) or any(not c.strip() for c in header):
                raise DataPrepError("CSV cần tiêu đề không rỗng, không trùng tên.")
            if len(header) > MAX_COLUMNS:
                raise DataPrepError(f"Vượt giới hạn {MAX_COLUMNS} cột.")
            count = 0
            for row in rows:
                if not row:
                    continue
                count += 1
                if len(row) != len(header):
                    raise DataPrepError(f"Dòng CSV {rows.line_num}: số trường khác tiêu đề.")
                if count > MAX_ROWS:
                    raise DataPrepError(f"Vượt giới hạn {MAX_ROWS:,} dòng.")
            text.seek(0)
            return pd.read_csv(
                text,
                sep=delimiter,
                dtype="string",
                keep_default_na=False,
                na_values=["", *options.missing_tokens],
                on_bad_lines="error",
            )
        finally:
            text.detach()
    if ext in (".json", ".jsonl", ".ndjson"):
        return _json_records(stream, options, ext != ".json")
    if ext == ".xlsx":
        return pd.read_excel(
            stream,
            sheet_name=options.sheet,
            engine="openpyxl",
            dtype=object,
            keep_default_na=False,
            na_values=["", *options.missing_tokens],
            nrows=MAX_ROWS + 1,
        )
    if ext in (".db", ".sqlite", ".sqlite3"):
        return _sqlite(source, options.table)
    if ext == ".parquet":
        return pd.read_parquet(stream)
    raise DataPrepError("Định dạng chưa hỗ trợ. Dùng CSV, TSV, JSON/JSONL, XLSX, SQLite hoặc Parquet.")

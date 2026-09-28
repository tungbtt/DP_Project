import json
import sqlite3
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dataprep.charts import comparison_chart, distribution
from dataprep.errors import DataPrepError
from dataprep.io import LoadOptions, load_data, sqlite_tables
from dataprep.pipeline import parse_config, run_pipeline
from dataprep.profile import correlation, profile_data
from dataprep.report import export_bundle
from dataprep.serialization import dumps

ROOT = Path(__file__).resolve().parents[1]


def config(*steps):
    return {"version": 1, "steps": list(steps)}


def test_csv_preserves_ids_and_explicit_null_policy():
    ds = load_data(b"customer_id,value,label\n001,1,NA\n002,,\n", "a.csv")
    assert ds.frame.customer_id.tolist() == ["001", "002"]
    assert ds.frame.loc[0, "label"] == "NA"
    assert pd.isna(ds.frame.loc[1, "value"])
    assert ds.frame.loc[0, "value"] == 1


@pytest.mark.parametrize("payload", [b"x,x\n1,2", b"x,y\n1,2,3", b"x,y\n1", b"x,y\n", b""])
def test_bad_csv_rejected(payload):
    with pytest.raises(DataPrepError):
        load_data(payload, "bad.csv")


def test_locale_and_explicit_missing():
    ds = load_data(
        b"id;price;label\n001;1,25;NA\n002;2,50;ok",
        "a.csv",
        LoadOptions(delimiter=";", decimal=",", missing_tokens=["NA"]),
    )
    assert ds.frame.price.tolist() == [1.25, 2.5]
    assert pd.isna(ds.frame.loc[0, "label"])


def test_json_nested_and_arrays():
    ds = load_data(ROOT / "examples/records.json", options=LoadOptions(json_path="data.records"))
    assert "student.name" in ds.frame
    assert ds.frame.loc[0, "id"] == "001"
    assert json.loads(ds.frame.loc[0, "tags"]) == ["A", "B"]
    assert len(ds.frame) == 3


def test_json_lines_errors_have_line_number():
    with pytest.raises(DataPrepError, match="dòng 2"):
        load_data(b'{"a":1}\nnot json', "a.jsonl")
    assert len(load_data(b'{"a":1}\n{"a":2}', "a.jsonl").frame) == 2


def test_sqlite_read_only_and_quoted_table(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "order data" (id TEXT, amount REAL)')
        conn.execute('INSERT INTO "order data" VALUES ("001", 5)')
    conn.close()
    source = db.read_bytes()
    assert sqlite_tables(source) == ["order data"]
    result = load_data(db)
    assert result.frame.loc[0, "id"] == "001"
    assert db.read_bytes() == source
    with pytest.raises(DataPrepError):
        load_data(db, options=LoadOptions(table="x; DROP TABLE x"))


def test_excel_roundtrip(tmp_path):
    path = tmp_path / "test.xlsx"
    pd.DataFrame({"id": ["001", "002"], "value": [3, 4]}).to_excel(path, index=False)
    result = load_data(path)
    assert result.frame.id.tolist() == ["001", "002"]


def test_profile_counts_and_id_exclusion():
    frame = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [None] * 4})
    p = profile_data(frame)
    assert p["overview"]["missing_cells"] == 4
    assert p["overview"]["missing_pct"] == 25
    matrix = correlation(frame)
    assert "id" not in matrix
    assert matrix.loc["x", "y"] == pytest.approx(1)
    assert profile_data(pd.DataFrame({"sbd": ["01000001"]}))["roles"]["sbd"] == "id"


def test_atomic_failure_and_original_unchanged():
    frame = pd.DataFrame({"name": [" a ", "b"], "age": ["12", "bad"]})
    saved = frame.copy(deep=True)
    with pytest.raises(DataPrepError, match="Bước 2"):
        run_pipeline(
            frame,
            config(
                {"op": "normalize_text", "columns": ["name"]},
                {"op": "cast", "columns": ["age"], "dtype": "numeric"},
            ),
        )
    assert_frame_equal(frame, saved)


def test_cleaning_order_logs_and_reproducibility():
    df = pd.DataFrame({"city": [" A ", "a", "b"], "v": [1.0, 1.0, np.nan]})
    cfg = config(
        {"op": "normalize_text", "columns": ["city"], "case": "lower"},
        {"op": "drop_duplicates", "columns": []},
        {"op": "fill_missing", "columns": ["v"], "strategy": "median"},
    )
    a, b = run_pipeline(df, cfg), run_pipeline(df, cfg)
    assert_frame_equal(a.frame, b.frame)
    assert a.log == b.log
    assert len(a.frame) == 2
    assert a.log[0]["changed_cells_retained"] == 1
    assert a.log[1]["removed_rows"] == 1
    assert a.log[2]["missing_after"] == 0


def test_datetime_invalid_and_unambiguous_format():
    frame = pd.DataFrame({"date": ["01/02/2026", "31/02/2026"]})
    step = {"op": "cast", "columns": ["date"], "dtype": "datetime", "format": "%d/%m/%Y", "errors": "coerce"}
    result = run_pipeline(frame, config(step))
    assert result.frame.loc[0, "date"].month == 2
    assert pd.isna(result.frame.loc[1, "date"])
    assert result.log[0]["missing_after"] == 1


@pytest.mark.parametrize("strategy", ["median", "mean", "mode"])
def test_all_missing_cannot_invent_statistic(strategy):
    with pytest.raises(DataPrepError, match="toàn giá trị thiếu"):
        run_pipeline(
            pd.DataFrame({"x": [None, None]}),
            config({"op": "fill_missing", "columns": ["x"], "strategy": strategy}),
        )


def test_outliers_range_and_deleted_column_failure():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0], "y": [1, 2, 3, 4, 5]})
    result = run_pipeline(frame, config({"op": "outliers", "columns": ["x"], "method": "clip"}))
    assert result.frame.x.max() == 7
    ranged = run_pipeline(
        frame, config({"op": "range", "columns": ["x"], "min": 0, "max": 5, "action": "drop"})
    )
    assert len(ranged.frame) == 4
    with pytest.raises(DataPrepError, match="Bước 2"):
        run_pipeline(
            frame, config({"op": "drop_columns", "columns": ["x"]}, {"op": "fill_missing", "columns": ["x"]})
        )


def test_schema_and_unknown_parameters_rejected():
    with pytest.raises(DataPrepError):
        run_pipeline(pd.DataFrame({"x": [1]}), {"expected_columns": ["other"], "steps": []})
    with pytest.raises(DataPrepError):
        parse_config('{"steps":[{"op":"fill_missing","columns":["x"],"stratgey":"mean"}]}')


def test_finite_numeric_cast_and_no_silent_mixed_fill():
    frame = pd.DataFrame({"x": ["1", "inf", "bad", None]})
    with pytest.raises(DataPrepError):
        run_pipeline(frame, config({"op": "fill_missing", "columns": ["x"], "strategy": "median"}))
    result = run_pipeline(frame, config({"op": "cast", "columns": ["x"], "errors": "coerce"}))
    assert result.frame.x.isna().sum() == 3


def test_empty_result_report_and_html_escaping():
    frame = pd.DataFrame({"x": [None], "<script>alert(1)</script>": [None]})
    result = run_pipeline(frame, config({"op": "drop_missing", "columns": []}))
    html, bundle = export_bundle(frame, result, '<img src=x onerror="alert(1)">')
    assert '<img src=x onerror="alert(1)">' not in html
    assert "&lt;img" in html
    assert "<script>alert(1)</script>" not in html
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {
            "report.html",
            "cleaned_data.csv",
            "pipeline.json",
            "processing_log.json",
            "schema.json",
            "source_metadata.json",
        }
        assert json.loads(archive.read("pipeline.json"))["steps"]


def test_example_pipeline_end_to_end():
    ds = load_data(ROOT / "examples/sales_dirty.csv")
    cfg = parse_config((ROOT / "examples/pipeline.json").read_text(encoding="utf-8"))
    result = run_pipeline(ds.frame, cfg)
    assert len(result.frame) == 20
    assert result.frame.age.max() <= 120
    assert result.frame.income.max() < 100_000_000
    assert result.frame.customer_id.iloc[0] == "001"
    assert result.frame.order_date.isna().sum() == 1
    html, _ = export_bundle(ds.frame, result, ds.name, ds.metadata)
    assert "Plotly.newPlot" in html
    assert 'src="https://cdn.plot.ly' not in html
    assert html.count("plotly.js v") == 1


def test_strict_json_no_nonfinite_literals():
    assert json.loads(dumps({"x": np.nan, "y": np.inf, "z": pd.NA})) == {"x": None, "y": None, "z": None}


def test_histogram_counts_and_shared_comparison_bins():
    frame = pd.DataFrame({"x": [1.0, 1.0, 1.0, 2.0, 1e12, np.nan]})
    figure = distribution(frame, "x", "numeric")
    assert sum(figure.data[0].y) == 5
    assert len(figure.data[0].y) <= 60
    compared = comparison_chart(frame, frame.iloc[:4], "x")
    np.testing.assert_array_equal(compared.data[0].x, compared.data[1].x)
    assert sum(compared.data[0].y) == 5
    assert sum(compared.data[1].y) == 4


def test_replace_constant_drop_missing_and_no_original_mutation():
    frame = pd.DataFrame({"city": ["HCM", "HN", None], "x": [None, 2.0, 3.0], "other": [None, None, "v"]})
    original = frame.copy(deep=True)
    result = run_pipeline(
        frame,
        config(
            {"op": "replace_values", "columns": ["city"], "mapping": {"HCM": "Ho Chi Minh"}},
            {"op": "fill_missing", "columns": ["x"], "strategy": "constant", "value": "5"},
            {"op": "drop_missing", "columns": ["city"], "how": "any"},
            {"op": "drop_columns", "columns": ["other"]},
        ),
    )
    assert result.frame.city.tolist() == ["Ho Chi Minh", "HN"]
    assert result.frame.x.tolist() == [5.0, 2.0]
    assert_frame_equal(frame, original)


def test_parquet_input(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    pd.DataFrame({"id": ["001"], "x": [5]}).to_parquet(path)
    assert load_data(path).frame.id.iloc[0] == "001"


def test_json_flatten_collision_rejected():
    with pytest.raises(DataPrepError, match="trùng sau khi làm phẳng"):
        load_data(b'[{"a.b": 1, "a": {"b": 2}}]', "collision.json")


def test_leading_zero_after_whitespace_is_preserved():
    result = load_data(b"reference\n 001\n 002", "id.csv", LoadOptions(delimiter=","))
    assert result.frame.reference.tolist() == [" 001", " 002"]

"""Small boundary fixtures for the large-data paths; full benchmark is opt-in."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import dataprep.io as ingestion
from dataprep.charts import correlation_chart
from dataprep.errors import DataPrepError
from dataprep.pipeline import changed_cells, run_pipeline
from dataprep.profile import correlation, resolve_roles
from dataprep.report import export_bundle


def test_two_million_limit_and_csv_boundary(monkeypatch):
    assert ingestion.MAX_ROWS == 2_000_000
    assert ingestion.MAX_BYTES == 512 * 1024**2
    monkeypatch.setattr(ingestion, "MAX_ROWS", 2)
    assert len(ingestion.load_data(b"x\n1\n2\n").frame) == 2
    with pytest.raises(DataPrepError, match="giới hạn"):
        ingestion.load_data(b"x\n1\n2\n3\n")


def test_path_csv_does_not_read_whole_file_to_bytes(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text('record_id,value\n001,"first\nsecond"\n002,other\n', encoding="utf-8", newline="")

    def disallow_bytes(*args):
        raise AssertionError("Path ingestion must stream from the file")

    monkeypatch.setattr(Path, "read_bytes", disallow_bytes)
    result = ingestion.load_data(path)
    assert result.frame.record_id.tolist() == ["001", "002"]
    assert result.frame.value.iloc[0] == "first\nsecond"
    assert result.metadata["memory_bytes"] > 0


def test_jsonl_chunk_union_and_late_errors(monkeypatch):
    monkeypatch.setattr(ingestion, "CHUNK_ROWS", 1)
    payload = b'{"a":1}\n{"b":{"x":2}}\n'
    result = ingestion.load_data(payload, "data.jsonl")
    assert list(result.frame) == ["a", "b.x"]
    assert pd.isna(result.frame.loc[0, "b.x"])
    assert result.frame.loc[1, "b.x"] == 2
    with pytest.raises(DataPrepError, match="dòng 3"):
        ingestion.load_data(payload + b"{invalid}\n", "data.jsonl")
    monkeypatch.setattr(ingestion, "MAX_ROWS", 1)
    with pytest.raises(DataPrepError, match="giới hạn"):
        ingestion.load_data(payload, "data.jsonl")


def test_explicit_roles_skip_inference(monkeypatch):
    def fail(*args):
        raise AssertionError("Explicit role must not trigger scanning")

    monkeypatch.setattr("dataprep.profile.infer_role", fail)
    assert resolve_roles(pd.DataFrame({"x": [1]}), {"x": "numeric"}) == {"x": "numeric"}


def test_correlation_bounds_input_before_calculation(monkeypatch):
    frame = pd.DataFrame({f"x{i}": range(100) for i in range(40)})
    original_corr = pd.DataFrame.corr
    seen = []

    def spy(self, *args, **kwargs):
        seen.append(self.shape)
        return original_corr(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "corr", spy)
    first = correlation(frame, max_rows=10, max_columns=3)
    second = correlation(frame, max_rows=10, max_columns=3)
    pd.testing.assert_frame_equal(first, second)
    assert seen == [(10, 3), (10, 3)]
    assert first.attrs == {"rows_used": 10, "total_rows": 100}
    monkeypatch.setattr("dataprep.charts.MAX_CORRELATION_ROWS", 10)
    chart = correlation_chart(frame)
    assert seen[-1] == (10, 30)
    assert "mẫu 10/100" in chart.layout.title.text


def test_change_count_across_chunk_boundary_and_removed_rows():
    before = pd.DataFrame({"x": pd.Series(["a"] * 50_002, dtype="string")})
    after = before.copy()
    after.loc[49_999, "x"] = pd.NA
    after.loc[50_001, "x"] = "b"
    after = after.drop(index=0)
    assert changed_cells(before, after) == 2


def test_shallow_snapshots_preserve_each_step_and_csv_zip():
    original = pd.DataFrame({"x": [1.0, None, 3.0], "s": [" a ", "b", "c"]})
    config = {
        "version": 1,
        "steps": [
            {"op": "fill_missing", "columns": ["x"], "strategy": "constant", "value": 2},
            {"op": "normalize_text", "columns": ["s"]},
        ],
    }
    result = run_pipeline(original, config)
    assert [item["changed_cells_retained"] for item in result.log] == [1, 1]
    assert pd.isna(original.loc[1, "x"])
    assert original.loc[0, "s"] == " a "
    _, payload = export_bundle(original, result)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv = archive.read("cleaned_data.csv")
        assert csv.startswith(b"\xef\xbb\xbf")
        restored = pd.read_csv(io.BytesIO(csv))
        assert restored.x.tolist() == [1, 2, 3]
        assert restored.s.tolist() == ["a", "b", "c"]

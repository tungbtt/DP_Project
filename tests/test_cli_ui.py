import os
import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dataprep.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_artifacts_and_overwrite_guard(tmp_path):
    args = [
        str(ROOT / "examples/sales_dirty.csv"),
        "--pipeline",
        str(ROOT / "examples/pipeline.json"),
        "--output",
        str(tmp_path),
    ]
    assert main(args) == 0
    assert (tmp_path / "result.zip").stat().st_size > 1000
    assert main(args) == 2


def test_cli_subprocess_with_vietnamese_path_and_legacy_console(tmp_path):
    target = tmp_path / "báo_cáo"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataprep",
            str(ROOT / "examples/records.json"),
            "--json-path",
            "data.records",
            "--output",
            str(target),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "báo_cáo" in result.stdout.decode("utf-8")
    assert (target / "report.html").exists()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_streamlit_demo_preview_apply_export():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert not app.exception
    assert len(app.metric) == 4
    first_profile = app.session_state["profiles"]["original"]
    app.run()
    assert app.session_state["profiles"]["original"] is first_profile
    editor = next(t for t in app.text_area if t.label == "Cấu hình JSON")
    editor.set_value((ROOT / "examples/pipeline.json").read_text(encoding="utf-8")).run()
    button(app, "Nạp cấu hình JSON").click().run()
    assert not app.exception
    button(app, "Xem trước toàn bộ quy trình").click().run()
    assert not app.exception
    assert len(app.session_state["preview"].frame) == 20
    button(app, "Áp dụng kết quả đã xem trước").click().run()
    assert not app.exception
    assert app.session_state["profiles"]["processed"]["overview"]["rows"] == 20
    button(app, "Tạo báo cáo và gói kết quả").click().run()
    assert not app.exception
    assert "artifacts" in app.session_state
    # Editing the pipeline must invalidate old applied/exported data.
    button(app, "Xóa bước").click().run()
    assert not app.exception
    assert "result" not in app.session_state
    assert "artifacts" not in app.session_state
    assert "processed" not in app.session_state["profiles"]


def test_streamlit_ml_preparation_and_stale_export_invalidation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert not app.exception
    button(app, "Chuẩn bị dữ liệu ML").click().run()
    assert not app.exception
    assert "ml_result" in app.session_state
    assert app.session_state["ml_result"].manifest["fit_partition"] == "train"
    assert app.session_state["ml_bundle"]
    scaler = next(s for s in app.selectbox if s.label == "Chuẩn hóa cột số")
    scaler.select("robust").run()
    assert not app.exception
    assert "ml_result" not in app.session_state
    assert "ml_bundle" not in app.session_state

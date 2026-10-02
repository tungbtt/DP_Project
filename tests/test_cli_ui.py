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


def navigate(app, stage):
    app.radio(key="workspace_stage").set_value(stage).run()
    assert not app.exception


def test_streamlit_demo_preview_apply_export():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert not app.exception
    assert len(app.metric) == 4
    first_profile = app.session_state["profiles"]["original"]
    app.run()
    assert app.session_state["profiles"]["original"] is first_profile
    navigate(app, "Làm sạch")
    editor = next(t for t in app.text_area if t.label == "Cấu hình JSON")
    editor.set_value((ROOT / "examples/pipeline.json").read_text(encoding="utf-8")).run()
    button(app, "Nạp cấu hình JSON").click().run()
    assert not app.exception
    button(app, "Xem trước toàn bộ quy trình").click().run()
    assert not app.exception
    assert len(app.session_state["preview"].frame) == 20
    button(app, "Áp dụng kết quả đã xem trước").click().run()
    assert not app.exception
    navigate(app, "Kết quả")
    next(r for r in app.radio if r.label == "Nội dung kết quả").set_value("So sánh trước – sau").run()
    assert app.session_state["profiles"]["processed"]["overview"]["rows"] == 20
    next(r for r in app.radio if r.label == "Nội dung kết quả").set_value("Tải kết quả").run()
    button(app, "Tạo báo cáo và gói kết quả").click().run()
    assert not app.exception
    assert "artifacts" in app.session_state
    # Editing the pipeline must invalidate old applied/exported data.
    navigate(app, "Làm sạch")
    button(app, "Xóa bước").click().run()
    assert not app.exception
    assert "result" not in app.session_state
    assert "artifacts" not in app.session_state
    assert "processed" not in app.session_state.get("profiles", {})


def test_streamlit_ml_preparation_and_stale_export_invalidation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert not app.exception
    navigate(app, "Học máy")
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
    navigate(app, "Khám phá")
    navigate(app, "Học máy")
    assert next(s for s in app.selectbox if s.label == "Chuẩn hóa cột số").value == "robust"


def test_navigation_without_dataset_and_lazy_charts(monkeypatch):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    navigate(app, "Kết quả")
    assert any("Nạp dữ liệu" in info.value for info in app.info)
    button(app, "Chọn nguồn dữ liệu").click().run()
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert not app.exception

    def unexpected(*args, **kwargs):
        raise AssertionError("Inactive EDA must not build charts")

    monkeypatch.setattr("dataprep.charts.scatter_chart", unexpected)
    monkeypatch.setattr("dataprep.charts.correlation_chart", unexpected)
    navigate(app, "Làm sạch")
    navigate(app, "Kết quả")
    navigate(app, "Nguồn dữ liệu")
    assert "dataset" in app.session_state
    button(app, "Đặt lại phiên làm việc").click().run()
    assert not app.exception
    assert "dataset" not in app.session_state


def test_remote_ui_download_select_load_and_failure_preserves_dataset(monkeypatch, tmp_path):
    import zipfile

    from dataprep.errors import DataPrepError
    from dataprep.remote import fetch_remote

    def download(source, target, *args):
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("first.csv", "id,x\n001,10\n002,20\n")
            archive.writestr("second.csv", "id,x\n003,30\n")
        return "sample.zip"

    monkeypatch.setattr("dataprep.remote._download", download)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio(key="source_kind").set_value("Kaggle").run()
    app.text_input(key="url_Kaggle").set_value("https://www.kaggle.com/datasets/owner/sample").run()
    app.text_input(key="kaggle_token").set_value("temporary-secret").run()
    button(app, "Tải danh sách file").click().run()
    assert not app.exception
    assert not app.text_input(key="kaggle_token").value
    app.selectbox(key="remote_member").select("second.csv").run()
    button(app, "Nạp dữ liệu").click().run()
    assert not app.exception
    assert app.session_state["dataset"].frame["id"].tolist() == ["003"]
    assert app.session_state["dataset"].metadata["remote"]["provider"] == "Kaggle"
    prior_key = app.session_state["dataset_key"]
    navigate(app, "Nguồn dữ liệu")
    app.radio(key="source_kind").set_value("Google Drive").run()
    assert any("Bất kỳ ai" in c.value for c in app.caption)
    app.text_input(key="url_Google Drive").set_value("https://drive.google.com/file/d/abcdefghijk/view").run()

    def fail(*args, **kwargs):
        raise DataPrepError("Không có quyền tải.")

    monkeypatch.setattr("dataprep.source_ui.fetch_remote", fail)
    button(app, "Tải danh sách file").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["dataset_key"] == prior_key
    assert button(app, "Nạp dữ liệu").disabled
    # New data invalidates previous results even with the same filename/first rows.
    monkeypatch.setattr("dataprep.source_ui.fetch_remote", fetch_remote)
    button(app, "Dùng dữ liệu mẫu").click().run()
    assert app.session_state["dataset_key"] != prior_key
    navigate(app, "Nguồn dữ liệu")
    path = app.session_state["remote_download"].path
    button(app, "Đặt lại phiên làm việc").click().run()
    assert not path.exists()

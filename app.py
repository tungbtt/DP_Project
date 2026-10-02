"""Data Prep Studio: one active workspace step per rerun."""

from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from dataprep.charts import (
    box_chart,
    comparison_chart,
    correlation_chart,
    distribution,
    missing_chart,
    scatter_chart,
)
from dataprep.errors import DataPrepError
from dataprep.ml_ui import render_ml
from dataprep.pipeline import OPERATIONS, parse_config, run_pipeline
from dataprep.profile import ROLES, compare_profiles, profile_data, resolve_roles
from dataprep.report import export_bundle
from dataprep.serialization import dumps
from dataprep.source_ui import clear_credentials, discard_download, render_source

ROOT = Path(__file__).parent
STAGES = ["Nguồn dữ liệu", "Khám phá", "Làm sạch", "Kết quả", "Học máy"]
st.set_page_config(page_title="Data Prep Studio", page_icon="📊", layout="wide")
st.markdown(
    """<style>
.block-container {max-width:1250px;padding-top:4.5rem;padding-bottom:3rem}
h1 {font-size:2rem!important;letter-spacing:-.04em}
h2,h3 {letter-spacing:-.02em}
[data-testid="stMetric"] {background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px}
[data-testid="stMetricLabel"] {color:#526577}
[data-testid="stSidebar"] {border-right:1px solid #e2e8f0}
[data-testid="stSidebar"] .stRadio label {padding:5px 0}
[data-testid="stExpander"] {background:#fff;border-radius:10px}
.brand {font-size:1.2rem;font-weight:750;color:#123e47;margin-bottom:4px}
.eyebrow {font-size:.72rem;letter-spacing:.14em;color:#527578;margin-bottom:12px}
</style>""",
    unsafe_allow_html=True,
)


def clear_results():
    for key in (
        "preview",
        "preview_config",
        "result",
        "artifacts",
        "ml_result",
        "ml_bundle",
        "ml_signature",
        "profiles",
    ):
        st.session_state.pop(key, None)


def set_config(config):
    st.session_state.config = config
    st.session_state.config_rev = st.session_state.get("config_rev", 0) + 1
    clear_results()


def load_into_session(dataset):
    st.session_state.dataset = dataset
    for key in list(st.session_state):
        if key.startswith(("ml_", "eda_", "roles_")):
            st.session_state.pop(key, None)
    st.session_state.dataset_key = uuid4().hex[:12]
    set_config(
        {
            "version": 1,
            "expected_columns": list(dataset.frame.columns),
            "roles": resolve_roles(dataset.frame),
            "steps": [],
        }
    )


def current_roles(frame, processed=False):
    cfg = st.session_state.config
    roles = {c: r for c, r in cfg.get("roles", {}).items() if c in frame}
    if processed:
        for step in cfg.get("steps", []):
            if step["op"] == "cast":
                for c in step["columns"]:
                    if c in frame and roles.get(c) not in ("id", "ignore"):
                        roles[c] = {"numeric": "numeric", "datetime": "datetime", "string": "text"}[
                            step.get("dtype", "numeric")
                        ]
    return resolve_roles(frame, roles)


def session_profile(frame, processed=False):
    # Cache compact summaries in this session, without hashing/caching the entire table.
    # Dataset/config changes clear this cache; applying a result invalidates its entry.
    profiles = st.session_state.setdefault("profiles", {})
    key = "processed" if processed else "original"
    if key not in profiles:
        with st.spinner("Đang tính thống kê trên toàn bộ dữ liệu…"):
            profiles[key] = profile_data(frame, current_roles(frame, processed))
    return profiles[key]


def go_to(stage):
    st.session_state.pending_stage = stage
    st.rerun()


def render_roles():
    st.subheader("Xem trước dữ liệu gốc")
    st.dataframe(original.head(100), use_container_width=True)
    st.caption("Hiển thị tối đa 100 dòng; các thống kê sử dụng toàn bộ dữ liệu đã nạp.")
    st.subheader("Vai trò cột")
    st.caption(
        "Vai trò quyết định thống kê và biểu đồ; không tự thay đổi kiểu lưu trữ. ID/ignore được loại khỏi tương quan."
    )
    roles = resolve_roles(original, config.get("roles"))
    schema = pd.DataFrame(
        {
            "Cột": list(original.columns),
            "Kiểu lưu trữ": [str(original[c].dtype) for c in original],
            "Vai trò": [roles[c] for c in original],
        }
    )
    edited = st.data_editor(
        schema,
        hide_index=True,
        use_container_width=True,
        disabled=["Cột", "Kiểu lưu trữ"],
        column_config={"Vai trò": st.column_config.SelectboxColumn(options=list(ROLES), required=True)},
        key=f"roles_{st.session_state.dataset_key}_{st.session_state.config_rev}",
    )
    if st.button("Lưu vai trò cột"):
        updated = dict(config)
        updated["roles"] = dict(zip(edited["Cột"], edited["Vai trò"]))
        set_config(updated)
        st.rerun()


def render_explore():
    st.title("Hiểu dữ liệu của bạn")
    st.caption("Thống kê trên toàn bộ bảng. Chọn một góc nhìn để giữ màn hình gọn và tập trung.")
    use_clean = st.checkbox("Khám phá dữ liệu đã làm sạch", disabled=active_result is None, key="eda_clean")
    processed = use_clean and active_result is not None
    data = active_result.frame if processed else original
    mapping = current_roles(data, processed)
    profile = session_profile(data, processed)
    metrics = st.columns(4)
    metrics[0].metric("Dòng dữ liệu", f"{len(data):,}")
    metrics[1].metric("Số cột", len(data.columns))
    metrics[2].metric("Ô thiếu", f"{profile['overview']['missing_cells']:,}")
    metrics[3].metric("Bản sao dư", f"{profile['overview']['duplicate_rows']:,}")
    view = st.radio("Góc nhìn", ["Tổng quan", "Theo cột", "Mối liên hệ"], horizontal=True, key="eda_view")
    if view == "Tổng quan":
        st.subheader("Chất lượng dữ liệu")
        if profile["issues"]:
            st.dataframe(
                pd.DataFrame(profile["issues"]).rename(
                    columns={
                        "column": "Cột",
                        "kind": "Loại",
                        "count": "Số lượng",
                        "message": "Phát hiện",
                        "suggestion": "Đề xuất",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("Không phát hiện vấn đề theo các quy tắc hiện có.")
        st.plotly_chart(missing_chart(data), use_container_width=True, key="missing_chart")
        with st.expander("Xem bảng gốc và điều chỉnh vai trò cột"):
            render_roles()
    elif view == "Theo cột":
        initial_column = next(
            (i for i, c in enumerate(data.columns) if mapping[c] not in ("id", "ignore")), 0
        )
        col = st.selectbox("Cột cần phân tích", list(data.columns), index=initial_column, key="eda_column")
        st.plotly_chart(distribution(data, col, mapping[col]), use_container_width=True, key="distribution")
        if mapping[col] == "numeric":
            st.plotly_chart(box_chart(data, col), use_container_width=True, key="box")
        with st.expander("Thống kê chi tiết của cột"):
            st.json(next(c for c in profile["columns"] if c["name"] == col))
    else:
        kind = st.radio("Biểu đồ", ["Tương quan", "Phân tán"], horizontal=True)
        if kind == "Tương quan":
            method = st.radio("Hệ số tương quan", ["pearson", "spearman"], horizontal=True)
            figure = correlation_chart(data, mapping, method)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True, key="correlation")
                st.caption("Toàn bộ cặp giá trị hợp lệ, ít nhất 3 cặp. Tương quan không chứng minh nhân quả.")
            else:
                st.info("Cần ít nhất hai cột có vai trò numeric để xem tương quan.")
        else:
            numeric = [c for c, role in mapping.items() if role == "numeric"]
            if len(numeric) >= 2:
                cols = st.columns(2)
                x = cols[0].selectbox("Scatter · trục X", numeric)
                y = cols[1].selectbox("Scatter · trục Y", [c for c in numeric if c != x])
                st.plotly_chart(scatter_chart(data, x, y), use_container_width=True, key="scatter")
                st.caption("Hiển thị toàn bộ cặp hợp lệ bằng WebGL; không lấy mẫu.")
            else:
                st.info("Cần ít nhất hai cột có vai trò numeric để vẽ phân tán.")
    st.divider()
    if st.button("Tiếp tục → Làm sạch", type="primary"):
        go_to("Làm sạch")


def render_clean():
    st.subheader("Quy trình làm sạch")
    st.caption(
        "Thực hiện theo thứ tự từ trên xuống trên bản sao. Mỗi lần xem trước đều chạy lại từ dữ liệu gốc."
    )
    labels = {
        "normalize_text": "Chuẩn hóa chuỗi",
        "replace_values": "Ánh xạ giá trị",
        "cast": "Chuyển kiểu",
        "fill_missing": "Điền dữ liệu thiếu",
        "drop_missing": "Xóa dòng thiếu",
        "drop_duplicates": "Loại bản ghi trùng",
        "drop_columns": "Xóa cột",
        "outliers": "Xử lý ngoại lệ IQR",
        "range": "Kiểm tra miền giá trị",
    }
    operation = st.selectbox("Thêm thao tác", list(OPERATIONS), format_func=lambda v: labels[v])
    with st.form("add_step"):
        selected = st.multiselect(
            "Cột áp dụng",
            list(original.columns),
            help="Để trống = toàn bộ cột, chỉ áp dụng với xóa dòng thiếu/loại trùng.",
        )
        step = {"op": operation, "columns": selected}
        if operation == "normalize_text":
            step["case"] = st.selectbox("Hoa/thường", ["keep", "lower", "upper", "casefold"])
            step["empty_as_missing"] = st.checkbox("Chuyển chuỗi rỗng sau trim thành thiếu", True)
        elif operation == "replace_values":
            mapping_text = st.text_area("Bảng ánh xạ JSON", '{"HCM": "Hồ Chí Minh", "TP.HCM": "Hồ Chí Minh"}')
        elif operation == "cast":
            step["dtype"] = st.selectbox("Kiểu đích", ["numeric", "datetime", "string"])
            step["errors"] = st.selectbox(
                "Giá trị lỗi",
                ["raise", "coerce"],
                format_func=lambda v: "Dừng và báo lỗi" if v == "raise" else "Chuyển thành thiếu",
            )
            step["format"] = st.text_input("Định dạng ngày (chỉ dùng với datetime)", "%d/%m/%Y")
            step["decimal"] = st.selectbox("Dấu thập phân khi chuyển số", [".", ","])
            step["thousands"] = st.text_input("Dấu hàng nghìn (để trống nếu không có)", max_chars=1)
        elif operation == "fill_missing":
            step["strategy"] = st.selectbox("Phương pháp điền", ["median", "mean", "mode", "constant"])
            step["value"] = st.text_input("Giá trị điền (chỉ dùng với constant)")
        elif operation == "drop_missing":
            step["how"] = st.selectbox(
                "Điều kiện",
                ["any", "all"],
                format_func=lambda v: (
                    "Thiếu ít nhất một cột đã chọn" if v == "any" else "Thiếu tất cả cột đã chọn"
                ),
            )
        elif operation == "drop_duplicates":
            step["keep"] = st.selectbox("Giữ bản ghi", ["first", "last", False], format_func=str)
        elif operation == "outliers":
            step["method"] = st.selectbox("Cách xử lý", ["clip", "missing", "drop"])
            step["factor"] = st.number_input("Hệ số IQR", min_value=0.1, value=1.5, step=0.1)
            st.caption("Ngoại lệ có thể hợp lệ. IQR không được áp dụng khi dưới 4 số hợp lệ hoặc IQR bằng 0.")
        elif operation == "range":
            step["min"] = st.number_input("Giá trị nhỏ nhất", value=0.0)
            step["max"] = st.number_input("Giá trị lớn nhất", value=120.0)
            step["action"] = st.selectbox("Nếu ngoài miền", ["missing", "drop"])
        submitted = st.form_submit_button("Thêm bước")
    if submitted:
        try:
            if operation == "replace_values":
                import json

                step["mapping"] = json.loads(mapping_text)
            updated = {**config, "steps": [*config["steps"], step]}
            parse_config(dumps(updated))
            set_config(updated)
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if config["steps"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Bước": i,
                        "Thao tác": labels[s["op"]],
                        "Cột": ", ".join(s.get("columns", [])) or "Tất cả",
                        "Tham số": dumps({k: v for k, v in s.items() if k not in ("op", "columns")}),
                    }
                    for i, s in enumerate(config["steps"], 1)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        controls = st.columns([2, 1, 1, 1])
        chosen = controls[0].selectbox("Bước cần điều chỉnh", list(range(1, len(config["steps"]) + 1))) - 1
        for button, delta, column in [("Lên", -1, controls[1]), ("Xuống", 1, controls[2])]:
            if column.button(button, disabled=not 0 <= chosen + delta < len(config["steps"])):
                steps = list(config["steps"])
                steps[chosen], steps[chosen + delta] = steps[chosen + delta], steps[chosen]
                set_config({**config, "steps": steps})
                st.rerun()
        if controls[3].button("Xóa bước"):
            set_config({**config, "steps": [s for i, s in enumerate(config["steps"]) if i != chosen]})
            st.rerun()
    else:
        st.info("Chưa có bước làm sạch. Bạn vẫn có thể xuất báo cáo EDA dữ liệu gốc.")
    with st.expander("Nhập / chỉnh sửa pipeline JSON"):
        imported = st.file_uploader("Tải pipeline.json", type=["json"], key="pipeline_upload")
        text = st.text_area(
            "Cấu hình JSON",
            value=dumps(config),
            height=260,
            key=f"pipeline_editor_{st.session_state.config_rev}",
        )
        if st.button("Nạp cấu hình JSON"):
            try:
                candidate = parse_config(imported.getvalue().decode("utf-8-sig") if imported else text)
                resolve_roles(original, candidate.get("roles"))
                candidate.setdefault("steps", [])
                set_config(candidate)
                st.rerun()
            except (DataPrepError, UnicodeError) as exc:
                st.error(str(exc))
    if st.button("Xem trước toàn bộ quy trình", type="primary"):
        try:
            with st.spinner("Đang chạy trên bản sao dữ liệu…"):
                st.session_state.preview = run_pipeline(original, config)
                st.session_state.preview_config = dumps(config)
        except DataPrepError as exc:
            st.session_state.pop("preview", None)
            st.error(str(exc))
    if "preview" in st.session_state:
        preview = st.session_state.preview
        st.success(f"Chạy thử thành công: {len(original):,} → {len(preview.frame):,} dòng.")
        st.dataframe(preview.frame.head(100), use_container_width=True)
        if preview.log:
            st.dataframe(
                pd.DataFrame(preview.log).drop(columns=["parameters", "dtype_changes"]),
                hide_index=True,
                use_container_width=True,
            )
        if st.button("Áp dụng kết quả đã xem trước", type="primary"):
            st.session_state.result = preview
            st.session_state.pop("artifacts", None)
            st.session_state.get("profiles", {}).pop("processed", None)
            st.rerun()
    st.divider()
    if st.button("Tiếp tục → Kết quả"):
        go_to("Kết quả")


def render_results():
    st.title("Kiểm tra và tải kết quả")
    view = st.radio("Nội dung kết quả", ["Tải kết quả", "So sánh trước – sau"], horizontal=True)
    if view == "Tải kết quả":
        st.subheader("Đóng gói kết quả")
        st.write(
            "Báo cáo HTML hoạt động offline, kèm dữ liệu CSV, pipeline, nhật ký, kiểu cột và thông tin nguồn."
        )
        if active_result is None:
            st.info("Chưa áp dụng làm sạch: báo cáo sẽ sử dụng dữ liệu gốc và pipeline rỗng.")
        st.caption("CSV không bảo toàn đầy đủ kiểu dữ liệu; schema.json lưu kiểu cột để đối chiếu.")
        if st.button("Tạo báo cáo và gói kết quả", type="primary"):
            try:
                with st.spinner("Đang tạo biểu đồ và báo cáo HTML…"):
                    result = active_result or run_pipeline(
                        original, {"version": 1, "roles": config.get("roles", {}), "steps": []}
                    )
                    html, bundle = export_bundle(original, result, dataset.name, dataset.metadata)
                    st.session_state.artifacts = {"html": html, "bundle": bundle, "result": result}
            except (DataPrepError, ValueError) as exc:
                st.error(str(exc))
        if "artifacts" in st.session_state:
            artifact = st.session_state.artifacts
            cols = st.columns(3)
            cols[0].download_button(
                "Tải báo cáo HTML", artifact["html"], "report.html", "text/html", use_container_width=True
            )
            cols[1].download_button(
                "Tải gói ZIP đầy đủ",
                artifact["bundle"],
                "data_prep_result.zip",
                "application/zip",
                use_container_width=True,
            )
            if len(artifact["result"].frame) <= 200_000:
                cols[2].download_button(
                    "Tải dữ liệu sạch CSV",
                    artifact["result"].frame.to_csv(index=False).encode("utf-8-sig"),
                    "cleaned_data.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                cols[2].info("CSV đầy đủ nằm trong gói ZIP. Tải ZIP và giải nén để lấy cleaned_data.csv.")
        st.download_button("Lưu pipeline đang cấu hình", dumps(config), "pipeline.json", "application/json")
    else:
        base_profile = session_profile(original)
        if active_result is None:
            st.info("Xem trước và áp dụng quy trình ở bước Làm sạch để so sánh.")
        else:
            before = base_profile
            after = session_profile(active_result.frame, True)
            st.dataframe(compare_profiles(before, after), hide_index=True, use_container_width=True)
            st.caption(
                "Tỷ lệ thiếu dùng tổng số ô của từng phiên bản. Xem cả số dòng/cột bị xóa khi đánh giá chất lượng."
            )
            comparable = [
                c for c in original if c in active_result.frame and current_roles(original)[c] == "numeric"
            ]
            if comparable:
                compare_col = st.selectbox("So sánh phân phối", comparable)
                fig = comparison_chart(original, active_result.frame, compare_col)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key="comparison")
            st.subheader("Vấn đề còn tồn tại")
            if after["issues"]:
                st.dataframe(pd.DataFrame(after["issues"]), hide_index=True, use_container_width=True)
            else:
                st.success("Không phát hiện vấn đề theo các quy tắc hiện có.")
    st.divider()
    if st.button("Chuẩn bị cho mô hình →"):
        go_to("Học máy")


clear_credentials()
if "pending_stage" in st.session_state:
    st.session_state.workspace_stage = st.session_state.pop("pending_stage")
# Preserve source/ML choices while their widgets are hidden. Credentials are excluded.
prefix = f"ml_{st.session_state.get('dataset_key', '')}_"
current_stage = st.session_state.get("workspace_stage", "Nguồn dữ liệu")
changed_stage = current_stage != st.session_state.get("_last_stage")
st.session_state._last_stage = current_stage
for state_key in list(st.session_state):
    preserve_ml = (current_stage != "Học máy" or changed_stage) and state_key.startswith(prefix)
    preserve_source = (current_stage != "Nguồn dữ liệu" or changed_stage) and (
        state_key.startswith(("url_", "remote_name_")) or state_key in ("source_kind", "remote_member")
    )
    if preserve_ml or preserve_source:
        st.session_state[state_key] = st.session_state[state_key]

with st.sidebar:
    st.markdown(
        '<div class="brand">◈ Data Prep Studio</div><div class="eyebrow">DỮ LIỆU RÕ RÀNG HƠN</div>',
        unsafe_allow_html=True,
    )
    stage = st.radio(
        "Các bước làm việc",
        STAGES,
        key="workspace_stage",
        format_func=lambda value: f"{STAGES.index(value) + 1:02d}  ·  {value}",
    )
    st.divider()
    if "dataset" in st.session_state:
        loaded = st.session_state.dataset
        st.caption("DỮ LIỆU ĐANG LÀM VIỆC")
        st.text(loaded.name)
        st.caption(f"{len(loaded.frame):,} dòng · {len(loaded.frame.columns)} cột")
        count = len(st.session_state.config["steps"])
        st.caption(
            f"{count} bước làm sạch · " + ("Đã áp dụng" if "result" in st.session_state else "Bản gốc")
        )
    else:
        st.caption("Chưa nạp dữ liệu. Bắt đầu ở bước 01.")
    if st.button("Đặt lại phiên làm việc", use_container_width=True):
        discard_download()
        for state_key in list(st.session_state):
            del st.session_state[state_key]
        st.rerun()
    st.caption("v1.3 · Python · Plotly")

if stage == "Nguồn dữ liệu":
    render_source(load_into_session, go_to, ROOT)
    st.stop()
if "dataset" not in st.session_state:
    st.title(stage)
    st.info("Nạp dữ liệu để bắt đầu bước này.")
    if st.button("Chọn nguồn dữ liệu", type="primary"):
        go_to("Nguồn dữ liệu")
    st.stop()

dataset = st.session_state.dataset
original = dataset.frame
config = st.session_state.config
active_result = st.session_state.get("result")
st.caption(f"BƯỚC {STAGES.index(stage) + 1:02d} / 05 · {dataset.name}")
if len(original) > 200_000:
    st.caption("Dữ liệu lớn: biểu đồ dùng toàn bộ dữ liệu hợp lệ, thời gian xử lý phụ thuộc RAM và số cột.")
if stage == "Khám phá":
    render_explore()
elif stage == "Làm sạch":
    st.title("Làm sạch có kiểm soát")
    render_clean()
elif stage == "Kết quả":
    render_results()
else:
    st.title("Sẵn sàng cho học máy")
    render_ml(original, current_roles(original), st.session_state.dataset_key)

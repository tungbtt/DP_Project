"""Vietnamese Streamlit UI. Run: streamlit run app.py"""

import hashlib
from pathlib import Path

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
from dataprep.io import MAX_BYTES, MAX_COLUMNS, MAX_ROWS, LoadOptions, load_data, sqlite_tables
from dataprep.ml_ui import render_ml
from dataprep.pipeline import OPERATIONS, parse_config, run_pipeline
from dataprep.profile import ROLES, compare_profiles, profile_data, resolve_roles
from dataprep.report import export_bundle
from dataprep.serialization import dumps

ROOT = Path(__file__).parent
st.set_page_config(page_title="Data Prep Studio", page_icon="📊", layout="wide")
st.markdown(
    """<style>
.block-container {max-width:1400px;padding-top:2rem}
[data-testid="stMetric"] {background:white;border:1px solid #dce4ed;border-radius:10px;padding:16px}
.hero {background:linear-gradient(110deg,#102a43,#125363);padding:28px 32px;border-radius:14px;color:white;margin-bottom:24px}
.hero p {color:#cce6ec;margin-bottom:0}.hero h1 {color:white;font-size:32px;padding-top:0}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="hero"><small>KHÁM PHÁ · LÀM SẠCH · KIỂM CHỨNG</small>
<h1>Data Prep Studio</h1><p>Hiểu dữ liệu của bạn. Kiểm soát từng thay đổi. Xuất báo cáo Plotly tương tác.</p></div>""",
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
    st.session_state.dataset_key = hashlib.sha256(
        (dataset.name + dataset.frame.head(20).to_csv()).encode()
    ).hexdigest()[:12]
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


with st.sidebar:
    st.subheader("Không gian dữ liệu")
    st.caption(
        f"Xử lý tại máy đang chạy ứng dụng. Giới hạn {MAX_BYTES // (1024 * 1024)} MiB, "
        f"{MAX_ROWS:,} dòng và {MAX_COLUMNS} cột. Khả năng xử lý phụ thuộc RAM và số cột."
    )
    uploaded = st.file_uploader(
        "Chọn dữ liệu",
        type=["csv", "tsv", "json", "jsonl", "ndjson", "xlsx", "db", "sqlite", "sqlite3", "parquet"],
        key="source",
    )
    with st.expander("Cấu hình đọc dữ liệu"):
        encoding = st.selectbox("Encoding", ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"])
        delimiter_label = st.selectbox("Dấu phân cách CSV", ["Tự phát hiện", ",", ";", "Tab", "|"])
        decimal = st.selectbox("Dấu thập phân", [".", ","])
        missing_text = st.text_input("Ký hiệu thiếu, cách nhau bằng dấu |", placeholder="NA|N/A|null")
        json_path = st.text_input("Nhánh JSON", placeholder="data.records")
        sheet_name = st.text_input("Tên sheet Excel", placeholder="Để trống: sheet đầu tiên")
        infer = st.checkbox("Nhận diện cột số khi mọi giá trị hợp lệ", value=True)
        st.caption("Chuỗi ngày giữ nguyên đến khi bạn chọn định dạng. Mã có số 0 đầu được giữ lại.")
    table = ""
    if uploaded and Path(uploaded.name).suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        try:
            tables = sqlite_tables(uploaded.getvalue())
            if tables:
                table = st.selectbox("Bảng SQLite", tables)
        except DataPrepError as exc:
            st.error(str(exc))
    if st.button("Nạp dữ liệu", type="primary", disabled=uploaded is None, use_container_width=True):
        try:
            options = LoadOptions(
                encoding=encoding,
                delimiter=None
                if delimiter_label == "Tự phát hiện"
                else "\t"
                if delimiter_label == "Tab"
                else delimiter_label,
                decimal=decimal,
                missing_tokens=[s for s in missing_text.split("|") if s],
                json_path=json_path,
                sheet=sheet_name or 0,
                table=table,
                infer_numeric=infer,
            )
            with st.spinner("Đang đọc và kiểm tra dữ liệu…"):
                load_into_session(load_data(uploaded.getvalue(), uploaded.name, options))
        except (DataPrepError, OSError) as exc:
            st.error(str(exc))
    st.divider()
    if st.button("Dùng dữ liệu mẫu", use_container_width=True):
        load_into_session(load_data(ROOT / "examples" / "sales_dirty.csv"))
        st.rerun()
    if st.button("Đặt lại phiên làm việc", use_container_width=True):
        for key in list(st.session_state):
            if key not in ("source",):
                del st.session_state[key]
        st.rerun()
    st.caption("Python · pandas · Plotly · Streamlit")

if "dataset" not in st.session_state:
    st.subheader("Bắt đầu từ dữ liệu của bạn")
    left, mid, right = st.columns(3)
    left.info("**1 · Khám phá**\n\nNạp CSV, JSON, Excel hoặc SQLite. Xem thống kê và vấn đề chất lượng.")
    mid.info("**2 · Làm sạch**\n\nChọn từng phép biến đổi, xem trước tác động rồi áp dụng.")
    right.info("**3 · Chia sẻ kết quả**\n\nTải dữ liệu sạch, pipeline tái sử dụng và báo cáo HTML offline.")
    st.caption("Dữ liệu mẫu có khoảng trắng thừa, bản ghi trùng, giá trị thiếu, sai kiểu và ngoại lệ.")
    st.stop()

dataset = st.session_state.dataset
original = dataset.frame
config = st.session_state.config
active_result = st.session_state.get("result")
base_profile = session_profile(original)
st.caption(
    f"Nguồn: {dataset.name} · {len(original):,} dòng × {len(original.columns)} cột · Bản gốc được giữ nguyên"
)
metrics = st.columns(4)
metrics[0].metric("Dòng dữ liệu", f"{len(original):,}")
metrics[1].metric("Số cột", len(original.columns))
metrics[2].metric("Ô thiếu", f"{base_profile['overview']['missing_cells']:,}")
metrics[3].metric("Bản sao dư", f"{base_profile['overview']['duplicate_rows']:,}")
if len(original) > 200_000:
    st.info(
        f"Bảng đang chiếm khoảng {base_profile['overview']['memory_bytes'] / 1024**2:,.0f} MiB trong RAM. "
        "Làm sạch, học máy và xuất kết quả cần thêm bộ nhớ. Thống kê dùng toàn bộ bảng; "
        "mọi biểu đồ dùng toàn bộ dữ liệu hợp lệ nên có thể mất thời gian với bảng lớn."
    )
tabs = st.tabs(
    ["1. Dữ liệu", "2. Khám phá EDA", "3. Làm sạch", "4. So sánh", "5. Xuất kết quả", "6. Học máy"]
)

with tabs[0]:
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

with tabs[1]:
    use_clean = st.checkbox("Khám phá dữ liệu đã làm sạch", disabled=active_result is None, key="eda_clean")
    data = active_result.frame if use_clean and active_result is not None else original
    mapping = current_roles(data, use_clean and active_result is not None)
    profile = session_profile(data, use_clean and active_result is not None)
    st.subheader("Chất lượng & đề xuất")
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
    col = st.selectbox("Cột cần phân tích", list(data.columns), key="eda_column")
    left, right = st.columns([2, 1])
    left.plotly_chart(distribution(data, col, mapping[col]), use_container_width=True, key="distribution")
    column_profile = next(c for c in profile["columns"] if c["name"] == col)
    right.json(column_profile)
    if mapping[col] == "numeric":
        st.plotly_chart(box_chart(data, col), use_container_width=True, key="box")
    method = st.radio("Hệ số tương quan", ["pearson", "spearman"], horizontal=True)
    figure = correlation_chart(data, mapping, method)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True, key="correlation")
        st.caption(
            "Tính trên toàn bộ các cặp giá trị hợp lệ, ít nhất 3 cặp. Tương quan không chứng minh nhân quả."
        )
    numeric = [c for c, role in mapping.items() if role == "numeric"]
    if len(numeric) >= 2:
        cols = st.columns(2)
        x = cols[0].selectbox("Scatter · trục X", numeric)
        y = cols[1].selectbox("Scatter · trục Y", [c for c in numeric if c != x])
        st.plotly_chart(scatter_chart(data, x, y), use_container_width=True, key="scatter")

with tabs[2]:
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

with tabs[3]:
    if active_result is None:
        st.info("Xem trước và áp dụng quy trình trong tab Làm sạch để so sánh.")
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

with tabs[4]:
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

with tabs[5]:
    render_ml(original, current_roles(original), st.session_state.dataset_key)

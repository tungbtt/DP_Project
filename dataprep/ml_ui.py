"""Streamlit controls for supervised ML preparation."""

from dataclasses import asdict

import pandas as pd
import streamlit as st

from .charts import distribution
from .errors import DataPrepError
from .ml import MLConfig, export_ml_bundle, prepare_ml
from .serialization import dumps


def render_ml(original, roles, dataset_key):
    st.subheader("Chuẩn bị dữ liệu cho học máy")
    st.info(
        "Dùng dữ liệu gốc đã nạp và chia tập trước khi điền thiếu, chuẩn hóa, mã hóa. "
        "Kết quả làm sạch ở tab 3 không tự đưa vào luồng ML vì có thể đã học thống kê trên toàn bộ dữ liệu."
    )
    st.caption(
        "Hỗ trợ bài toán có giám sát. Chọn feature phù hợp; loại mã định danh, nội dung tự do "
        "và các cột tiết lộ nhãn. Nếu cùng khách hàng xuất hiện nhiều lần, cần chia theo nhóm bên ngoài."
    )
    prefix = f"ml_{dataset_key}_"
    cols = st.columns(2)
    target = cols[0].selectbox(
        "Cột mục tiêu (target)",
        list(original.columns),
        index=len(original.columns) - 1,
        key=prefix + "target",
    )
    task = cols[1].selectbox(
        "Bài toán",
        ["regression", "classification"],
        format_func=lambda v: "Hồi quy" if v == "regression" else "Phân loại",
        key=prefix + "task",
    )
    candidates = [c for c in original if c != target]
    feature_cols = st.columns(2)
    numeric = feature_cols[0].multiselect(
        "Feature số",
        candidates,
        default=[c for c in candidates if roles[c] == "numeric"],
        key=prefix + target + "numeric",
    )
    categorical = feature_cols[1].multiselect(
        "Feature phân loại",
        [c for c in candidates if c not in numeric],
        default=[
            c
            for c in candidates
            if c not in numeric and roles[c] == "category" and original[c].nunique() <= 10
        ],
        key=prefix + target + "categorical",
    )
    excluded = [c for c in original if c not in numeric + categorical + [target]]
    st.caption("Các cột không dùng làm feature: " + (", ".join(excluded) or "Không có"))
    cols = st.columns(3)
    scaler_names = {
        "standard": "StandardScaler · trung bình 0, độ lệch chuẩn 1",
        "minmax": "MinMaxScaler · thang [0, 1] trên train",
        "robust": "RobustScaler · median và IQR",
        "none": "Giữ thang đo",
    }
    scaler = cols[0].selectbox(
        "Chuẩn hóa cột số", list(scaler_names), format_func=scaler_names.get, key=prefix + "scaler"
    )
    imputer = cols[1].selectbox("Điền thiếu cột số", ["median", "mean", "zero"], key=prefix + "imputer")
    encoding = cols[2].selectbox("Mã hóa phân loại", ["onehot", "ordinal"], key=prefix + "encoding")
    if encoding == "ordinal":
        st.caption(
            "Ordinal tạo các mã số không có thứ tự nghiệp vụ. One-hot phù hợp hơn khi các nhãn không có thứ tự."
        )
    st.caption(
        "Cột phân loại thiếu được gán nhãn riêng. Mọi tham số đều học trên train và dùng lại cho validation/test."
    )
    with st.expander("Chia tập và xử lý giá trị lỗi", expanded=True):
        cols = st.columns(3)
        test_pct = cols[0].number_input(
            "Test (%)", min_value=5, max_value=45, value=20, step=5, key=prefix + "test"
        )
        val_pct = cols[1].number_input(
            "Validation (%)", min_value=0, max_value=45, value=20, step=5, key=prefix + "val"
        )
        seed = cols[2].number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, value=42, key=prefix + "seed"
        )
        split_method = st.radio(
            "Cách chia tập",
            ["random", "time"],
            horizontal=True,
            format_func=lambda v: "Ngẫu nhiên" if v == "random" else "Theo thời gian",
            key=prefix + "split",
        )
        stratify = st.checkbox(
            "Giữ tỷ lệ các lớp khi chia ngẫu nhiên",
            value=True,
            disabled=task != "classification" or split_method != "random",
            key=prefix + "stratify",
        )
        time_column, time_format = "", "ISO8601"
        if split_method == "time":
            time_column = st.selectbox("Cột dùng để chia thời gian", candidates, key=prefix + "time_col")
            time_format = st.text_input("Định dạng thời gian", "%d/%m/%Y", key=prefix + "time_format")
            st.caption(
                "Train lấy các thời điểm sớm nhất. Cột chia thời gian phải được bỏ khỏi danh sách feature."
            )
        max_categories = st.number_input(
            "Tối đa nhóm mã hóa mỗi cột", min_value=2, max_value=200, value=50, key=prefix + "max_categories"
        )
        numeric_errors = st.selectbox(
            "Giá trị không hợp lệ trong feature số",
            ["raise", "coerce"],
            format_func=lambda v: "Dừng và báo lỗi" if v == "raise" else "Chuyển thành thiếu trước khi điền",
            key=prefix + "numeric_errors",
        )
        drop_missing = st.checkbox("Cho phép loại dòng thiếu target", value=False, key=prefix + "drop_target")
    cfg = MLConfig(
        target=target,
        numeric_columns=numeric,
        categorical_columns=categorical,
        task=task,
        scaler=scaler,
        numeric_imputer=imputer,
        encoding=encoding,
        max_categories=int(max_categories),
        test_size=test_pct / 100,
        validation_size=val_pct / 100,
        seed=int(seed),
        split_method=split_method,
        stratify=stratify,
        time_column=time_column,
        time_format=time_format,
        drop_missing_target=drop_missing,
        numeric_errors=numeric_errors,
    )
    signature = dumps(asdict(cfg))
    if st.session_state.get("ml_signature") != signature:
        st.session_state.pop("ml_result", None)
        st.session_state.pop("ml_bundle", None)
    if st.button("Chuẩn bị dữ liệu ML", type="primary"):
        st.session_state.pop("ml_result", None)
        st.session_state.pop("ml_bundle", None)
        try:
            with st.spinner("Chia tập và học bộ tiền xử lý trên train…"):
                result = prepare_ml(original, cfg)
                bundle = export_ml_bundle(result)
                st.session_state.ml_result = result
                st.session_state.ml_bundle = bundle
                st.session_state.ml_signature = signature
        except DataPrepError as exc:
            st.error(str(exc))
    if "ml_result" not in st.session_state:
        return
    result = st.session_state.ml_result
    st.success("Đã tạo các tập dữ liệu và bộ tiền xử lý có thể dùng lại cho dữ liệu mới.")
    columns = st.columns(4)
    for ui, split in zip(columns[:3], ("train", "validation", "test")):
        ui.metric(split.title(), result.matrices[split].shape[0])
    columns[3].metric("Feature sau mã hóa", len(result.feature_names))
    for note in result.manifest["notes"]:
        st.caption(note)
    st.write("20 dòng đầu của train sau biến đổi (tối đa 30 feature):")
    st.dataframe(
        pd.DataFrame(result.matrices["train"][:20, :30].toarray(), columns=result.feature_names[:30]),
        use_container_width=True,
    )
    if numeric:
        col = st.selectbox("So sánh thang đo trên train", numeric, key=prefix + "plot_col")
        position = numeric.index(col)
        before = original.iloc[result.row_indices["train"]][[col]]
        after = pd.DataFrame({col: result.matrices["train"][:, position].toarray().ravel()})
        charts = st.columns(2)
        charts[0].caption("Trước điền thiếu / chuẩn hóa · train")
        charts[0].plotly_chart(
            distribution(before, col, "numeric"), use_container_width=True, key="ml_before"
        )
        charts[1].caption("Sau điền thiếu / chuẩn hóa · train (thang đo mới)")
        charts[1].plotly_chart(distribution(after, col, "numeric"), use_container_width=True, key="ml_after")
    if result.manifest["target_mapping"]:
        st.write("Mã hóa nhãn mục tiêu:", result.manifest["target_mapping"])
    st.download_button(
        "Tải gói dữ liệu ML", st.session_state.ml_bundle, "ml_ready.zip", "application/zip", type="primary"
    )
    st.download_button("Lưu cấu hình ML", signature, "ml_config.json", "application/json")
    st.caption(
        "ZIP gồm X/y train–validation–test, vị trí dòng nguồn, preprocessor.joblib, cấu hình và hướng dẫn. "
        "Luôn có ma trận NPZ; chỉ thêm CSV nếu mỗi tập không quá 1 triệu ô."
    )

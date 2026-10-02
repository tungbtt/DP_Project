"""Source selection, remote download lifecycle and common parsing options."""

from pathlib import Path

import streamlit as st

from .errors import DataPrepError
from .io import MAX_BYTES, MAX_COLUMNS, MAX_ROWS, LoadOptions, load_data, sqlite_tables
from .remote import EXTENSIONS, fetch_remote, parse_source


def discard_download():
    remote = st.session_state.pop("remote_download", None)
    if remote is not None:
        remote.close()
    st.session_state.pop("remote_input", None)


def clear_credentials():
    # Called before credential widgets are instantiated on the next rerun.
    if st.session_state.pop("clear_credentials", False):
        for key in ("kaggle_token", "kaggle_username", "kaggle_key"):
            st.session_state.pop(key, None)


def render_source(on_load, go_to, root):
    st.title("Bắt đầu với dữ liệu")
    st.write("Chọn một nguồn, nạp bảng dữ liệu rồi khám phá và làm sạch theo từng bước.")
    source = st.radio(
        "Nguồn dữ liệu", ["Máy tính", "Kaggle", "Google Drive"], horizontal=True, key="source_kind"
    )
    uploaded, remote, member = None, None, ""
    if source == "Máy tính":
        uploaded = st.file_uploader("Chọn dữ liệu", type=sorted(ext[1:] for ext in EXTENSIONS), key="source")
    else:
        if source == "Kaggle":
            st.caption("Dán link trang dataset. Nếu ZIP có nhiều bảng, bạn sẽ chọn một file sau khi tải.")
            placeholder = "https://www.kaggle.com/datasets/uciml/iris"
        else:
            st.caption(
                "Chia sẻ file bằng quyền “Bất kỳ ai có đường liên kết” và cho phép tải xuống. Hỗ trợ file hoặc ZIP; chưa hỗ trợ thư mục và Google Sheets."
            )
            placeholder = "https://drive.google.com/file/d/FILE_ID/view"
        url = st.text_input("Link dữ liệu", placeholder=placeholder, key=f"url_{source}").strip()
        token = username = api_key = ""
        if source == "Kaggle":
            with st.expander("Xác thực Kaggle (khi dataset yêu cầu)"):
                st.caption(
                    "Lấy API token trong Settings → API của Kaggle. Thông tin xác thực được xóa khỏi ô nhập sau mỗi lần tải; không ghi vào báo cáo."
                )
                token = st.text_input("Kaggle API token", type="password", key="kaggle_token")
                st.caption("Nếu dùng API key cũ, nhập cả hai trường bên dưới thay cho token.")
                username = st.text_input("Kaggle username (legacy)", key="kaggle_username")
                api_key = st.text_input("Kaggle API key (legacy)", type="password", key="kaggle_key")
        if st.button("Tải danh sách file", type="primary", disabled=not url):
            feedback = st.empty()
            try:
                parsed = parse_source(url)
                if parsed.provider != source:
                    raise DataPrepError(f"Hãy nhập link {source} hoặc đổi nguồn dữ liệu.")

                def progress(done, total):
                    suffix = f" / {total / 1024**2:,.1f} MiB" if total else " MiB"
                    feedback.caption(f"Đang tải: {done / 1024**2:,.1f}" + suffix)

                with st.spinner("Đang kết nối và tải dữ liệu…"):
                    downloaded = fetch_remote(
                        url, token=token, username=username, key=api_key, progress=progress
                    )
                discard_download()
                st.session_state.remote_download = downloaded
                st.session_state.remote_input = url
                st.session_state.pop("remote_error", None)
            except DataPrepError as exc:
                st.session_state.remote_error = str(exc)
                st.session_state.remote_error_input = url
            finally:
                st.session_state.clear_credentials = True
            st.rerun()
        if st.session_state.get("remote_error") and st.session_state.get("remote_error_input") == url:
            st.error(st.session_state.remote_error)
        candidate = st.session_state.get("remote_download")
        if (
            candidate is not None
            and st.session_state.get("remote_input") == url
            and candidate.source.provider == source
        ):
            remote = candidate
            st.success(f"Đã tải {remote.path.stat().st_size:,} byte. Chọn file cần nạp.")
            member = st.selectbox("File trong nguồn đã tải", remote.files(), key="remote_member")

    ready = uploaded is not None or remote is not None
    filename = uploaded.name if uploaded is not None else Path(member).name
    options = LoadOptions()
    with st.expander("Tùy chọn đọc dữ liệu", expanded=False):
        cols = st.columns(3)
        options.encoding = cols[0].selectbox("Encoding", ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"])
        separator = cols[1].selectbox("Dấu phân cách CSV", ["Tự phát hiện", ",", ";", "Tab", "|"])
        options.delimiter = None if separator == "Tự phát hiện" else "\t" if separator == "Tab" else separator
        options.decimal = cols[2].selectbox("Dấu thập phân", [".", ","])
        missing = st.text_input("Ký hiệu thiếu, cách nhau bằng dấu |", placeholder="NA|N/A|null")
        options.missing_tokens = [v for v in missing.split("|") if v]
        cols = st.columns(2)
        options.json_path = cols[0].text_input("Nhánh JSON", placeholder="data.records")
        options.sheet = cols[1].text_input("Tên sheet Excel", placeholder="Để trống: sheet đầu tiên") or 0
        options.infer_numeric = st.checkbox("Nhận diện cột số khi mọi giá trị hợp lệ", value=True)
        if remote is not None:
            filename = st.text_input("Tên file và định dạng", value=filename, key=f"remote_name_{member}")
        st.caption("Mã có số 0 đầu được giữ nguyên. Ngày chỉ chuyển kiểu khi bạn chỉ định định dạng.")
    if ready and Path(filename).suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        try:
            payload = uploaded.getvalue() if uploaded is not None else remote.materialize(member)
            tables = sqlite_tables(payload)
            if tables:
                options.table = st.selectbox("Bảng SQLite", tables)
            else:
                st.error("SQLite không có bảng dữ liệu.")
                ready = False
        except (DataPrepError, OSError) as exc:
            st.error(str(exc))
            ready = False
    if st.button("Nạp dữ liệu", type="primary", disabled=not ready, use_container_width=True):
        try:
            with st.spinner("Đang đọc và kiểm tra toàn bộ dữ liệu…"):
                data = (
                    load_data(uploaded.getvalue(), uploaded.name, options)
                    if uploaded is not None
                    else remote.load(member, options, filename)
                )
            on_load(data)
            go_to("Khám phá")
        except (DataPrepError, OSError) as exc:
            st.error(str(exc))
    st.caption(
        f"Tối đa {MAX_ROWS:,} dòng · {MAX_COLUMNS} cột · {MAX_BYTES // 1024**2} MiB mỗi file hoặc gói tải. Dữ liệu xử lý trên máy chủ chạy ứng dụng."
    )
    st.divider()
    left, right = st.columns([2, 1])
    left.markdown("**Chưa có file? Thử quy trình với dữ liệu mẫu.**")
    left.caption("Bảng bán hàng gồm giá trị thiếu, bản ghi trùng và dữ liệu cần chuẩn hóa.")
    if right.button("Dùng dữ liệu mẫu", use_container_width=True):
        on_load(load_data(root / "examples" / "sales_dirty.csv"))
        go_to("Khám phá")

"""Bounded, provider-specific remote downloads; no arbitrary URL fetching."""

import ipaddress
import re
import socket
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import requests

from .errors import DataPrepError
from .io import MAX_BYTES, LoadOptions, load_data

EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".db", ".sqlite", ".sqlite3", ".parquet"}
MAX_FILES = 1000
DOWNLOAD_SECONDS = 300


@dataclass(frozen=True)
class RemoteSource:
    provider: str
    reference: str
    url: str
    download_url: str


def parse_source(url):
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise DataPrepError("Nhập link HTTPS hợp lệ từ Kaggle hoặc Google Drive.") from exc
    host = (parsed.hostname or "").lower()
    if host in {"kaggle.com", "www.kaggle.com"}:
        match = re.fullmatch(r"/datasets/([\w-]+)/([\w-]+)(?:/versions/(\d+))?/?", parsed.path, re.ASCII)
        if not match:
            raise DataPrepError(
                "Dùng link dataset Kaggle dạng https://www.kaggle.com/datasets/owner/name. Link competition/notebook chưa hỗ trợ."
            )
        owner, name, version = match.groups()
        canonical = f"https://www.kaggle.com/datasets/{owner}/{name}"
        download = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}"
        if version:
            canonical += f"/versions/{version}"
            download += "?" + urlencode({"datasetVersionNumber": version})
        return RemoteSource("Kaggle", f"{owner}/{name}", canonical, download)
    if host in {"drive.google.com", "docs.google.com"}:
        match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)(?:/(?:view|edit|preview))?/?", parsed.path)
        query = parse_qs(parsed.query)
        file_id = (
            match.group(1) if match else query.get("id", [""])[0] if parsed.path in ("/open", "/uc") else ""
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]{10,200}", file_id):
            raise DataPrepError(
                "Dùng link một file Google Drive. Thư mục và tài liệu Google Sheets/Docs chưa hỗ trợ; hãy xuất thành CSV/XLSX trước."
            )
        params = {"id": file_id, "export": "download"}
        resource_key = query.get("resourcekey", [""])[0]
        if resource_key:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", resource_key):
                raise DataPrepError("Resource key của link Google Drive không hợp lệ.")
            params["resourcekey"] = resource_key
        return RemoteSource(
            "Google Drive",
            file_id,
            f"https://drive.google.com/file/d/{file_id}/view",
            "https://drive.google.com/uc?" + urlencode(params),
        )
    raise DataPrepError("Chỉ hỗ trợ link dataset Kaggle hoặc file Google Drive.")


def _check_download_url(url, provider):
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        valid = (
            parsed.scheme == "https"
            and parsed.port in (None, 443)
            and not parsed.username
            and not parsed.password
        )
        if provider == "Kaggle":
            allowed = host in {"www.kaggle.com", "kaggle.com", "storage.googleapis.com"} or host.endswith(
                ".storage.googleapis.com"
            )
        else:
            allowed = host in {
                "drive.google.com",
                "docs.google.com",
                "drive.usercontent.google.com",
            } or host.endswith(".googleusercontent.com")
        if not valid or not allowed:
            raise DataPrepError("Nguồn chuyển hướng tới địa chỉ tải không được hỗ trợ.")
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise DataPrepError("Không cho phép tải từ địa chỉ mạng nội bộ.")
    except (ValueError, OSError) as exc:
        raise DataPrepError("Không phân giải được địa chỉ máy chủ dữ liệu.") from exc


class _DriveForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.active = "", {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.active = attrs.get("method", "get").lower() == "get"
            self.action = attrs.get("action", "") if self.active else ""
        if self.active and tag == "input" and attrs.get("type") == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def _confirmation_url(html, current, file_id):
    form = _DriveForm()
    form.feed(html)
    if not form.action:
        raise DataPrepError(
            "Google Drive chưa cho tải file. Hãy bật quyền 'Bất kỳ ai có đường liên kết', cho phép tải xuống và kiểm tra hạn mức tải."
        )
    target = urljoin(current, form.action)
    parts = urlsplit(target)
    fields = {k: v[0] for k, v in parse_qs(parts.query).items()}
    fields.update(form.fields)
    if fields.get("id") != file_id:
        raise DataPrepError("Trang xác nhận Google Drive không khớp file đã chọn.")
    return parts._replace(query=urlencode(fields)).geturl()


def _safe_name(name):
    # This is a display name only. Files are always written to generated local paths.
    return Path(name.replace("\\", "/")).name.replace("\x00", "")[:240]


def _download(source, target, token="", username="", key="", progress=None):
    started = time.monotonic()
    url = source.download_url
    headers = {"User-Agent": "DataPrepStudio/1.3", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = "Bearer " + token.strip()
    auth = (username, key) if username and key and not token else None
    with requests.Session() as session:
        # Do not inherit .netrc credentials, proxy credentials or browser cookies.
        session.trust_env = False
        for _ in range(8):
            _check_download_url(url, source.provider)
            if time.monotonic() - started > DOWNLOAD_SECONDS:
                raise DataPrepError("Quá thời gian tải. Thử lại hoặc tải file về máy trước.")
            host = urlsplit(url).hostname
            if host != "www.kaggle.com" or not urlsplit(url).path.startswith("/api/"):
                headers.pop("Authorization", None)
                auth = None
            with session.get(
                url, headers=headers, auth=auth, stream=True, allow_redirects=False, timeout=(15, 30)
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise DataPrepError("Máy chủ không cung cấp địa chỉ tải tiếp theo.")
                    url = urljoin(url, location)
                    continue
                if response.status_code in (401, 403):
                    raise DataPrepError(
                        "Không có quyền tải. Kaggle có thể cần API token hoặc chấp nhận điều khoản; Drive cần quyền chia sẻ và tải xuống."
                    )
                if response.status_code == 404:
                    raise DataPrepError("Không tìm thấy dữ liệu hoặc bạn không có quyền truy cập.")
                if response.status_code == 429:
                    raise DataPrepError("Nguồn dữ liệu đang giới hạn lượt tải. Vui lòng thử lại sau.")
                if response.status_code != 200:
                    raise DataPrepError(f"Máy chủ trả lỗi HTTP {response.status_code}. Vui lòng thử lại sau.")
                size_text = response.headers.get("Content-Length", "")
                total = int(size_text) if size_text.isdigit() else None
                if total and total > MAX_BYTES:
                    raise DataPrepError("File tải vượt giới hạn 512 MiB.")
                is_html = "text/html" in response.headers.get("Content-Type", "").lower()
                written = 0
                with target.open("wb") as output:
                    for chunk in response.iter_content(256 * 1024):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > (1024 * 1024 if is_html else MAX_BYTES):
                            raise DataPrepError("Nội dung tải vượt giới hạn cho phép.")
                        if time.monotonic() - started > DOWNLOAD_SECONDS:
                            raise DataPrepError("Quá thời gian tải dữ liệu.")
                        output.write(chunk)
                        if progress and not is_html:
                            progress(written, total)
                with target.open("rb") as check:
                    prefix = check.read(512).lstrip().lower()
                is_html = is_html or prefix.startswith((b"<!doctype html", b"<html"))
                if is_html:
                    if source.provider == "Google Drive" and written <= 1024 * 1024:
                        url = _confirmation_url(
                            target.read_text(encoding="utf-8", errors="replace"), url, source.reference
                        )
                        continue
                    raise DataPrepError(
                        "Nguồn trả về trang đăng nhập/xác minh thay vì file. Kiểm tra quyền truy cập hoặc tải file thủ công."
                    )
                if not written:
                    raise DataPrepError("File tải về rỗng.")
                if total is not None and written != total and not response.headers.get("Content-Encoding"):
                    raise DataPrepError("File tải chưa đầy đủ. Hãy thử lại.")
                msg = Message()
                msg["Content-Disposition"] = response.headers.get("Content-Disposition", "")
                return _safe_name(
                    msg.get_filename() or ("dataset.zip" if source.provider == "Kaggle" else "download")
                )
    raise DataPrepError("Nguồn chuyển hướng hoặc yêu cầu xác nhận quá nhiều lần.")


@dataclass
class RemoteDownload:
    source: RemoteSource
    name: str
    path: Path
    _temp: tempfile.TemporaryDirectory = field(repr=False)
    _selected: str = field(default="", repr=False)

    def is_archive(self):
        # XLSX is itself a ZIP container, but is a single supported data file.
        return Path(self.name).suffix.lower() != ".xlsx" and zipfile.is_zipfile(self.path)

    def close(self):
        self._temp.cleanup()

    def files(self):
        if not self.is_archive():
            return [self.name]
        with zipfile.ZipFile(self.path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES:
                raise DataPrepError("ZIP có quá nhiều file; tối đa 1.000 mục.")
            names = set()
            result = []
            for item in entries:
                path = PurePosixPath(item.filename.replace("\\", "/"))
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or ":" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)
                ):
                    raise DataPrepError("ZIP chứa đường dẫn hoặc liên kết không an toàn.")
                if item.filename in names:
                    raise DataPrepError("ZIP chứa tên file trùng nhau.")
                names.add(item.filename)
                if not item.is_dir() and path.suffix.lower() in EXTENSIONS:
                    if item.flag_bits & 1:
                        raise DataPrepError("ZIP có mật khẩu chưa được hỗ trợ.")
                    if item.file_size > MAX_BYTES:
                        continue
                    result.append(item.filename)
            if not result:
                raise DataPrepError("ZIP không có file dữ liệu được hỗ trợ trong giới hạn 512 MiB mỗi file.")
            return sorted(result)

    def materialize(self, member):
        if member not in self.files():
            raise DataPrepError("File được chọn không thuộc nguồn đã tải.")
        if not self.is_archive():
            return self.path
        target = Path(self._temp.name) / "selected-data"
        if self._selected == member and target.exists():
            return target
        self._selected = ""
        try:
            with (
                zipfile.ZipFile(self.path) as archive,
                archive.open(member) as incoming,
                target.open("wb") as outgoing,
            ):
                size = 0
                while chunk := incoming.read(256 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise DataPrepError("File sau giải nén vượt giới hạn 512 MiB.")
                    outgoing.write(chunk)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
            raise DataPrepError("Không giải nén được file đã chọn. ZIP có thể bị lỗi.") from exc
        self._selected = member
        return target

    def load(self, member, options=None, filename=None):
        name = _safe_name(filename or member)
        if Path(name).suffix.lower() not in EXTENSIONS:
            raise DataPrepError(
                "Không nhận diện được định dạng. Nhập tên file kèm đuôi như data.csv trong cấu hình đọc."
            )
        result = load_data(self.materialize(member), filename=name, options=options or LoadOptions())
        result.metadata["remote"] = {"provider": self.source.provider, "url": self.source.url, "file": member}
        return result


def fetch_remote(url, *, token="", username="", key="", progress=None):
    source = parse_source(url)
    if bool(username) != bool(key):
        raise DataPrepError("Kaggle legacy cần cả username và API key.")
    temp = tempfile.TemporaryDirectory(prefix="dataprep-download-")
    path = Path(temp.name) / "download"
    try:
        name = _download(source, path, token, username, key, progress)
        if zipfile.is_zipfile(path) and Path(name).suffix.lower() != ".xlsx":
            with zipfile.ZipFile(path) as archive:
                if {"[Content_Types].xml", "xl/workbook.xml"}.issubset(archive.namelist()):
                    name = "download.xlsx"
        result = RemoteDownload(source, name, path, temp)
        result.files()
        return result
    except DataPrepError:
        temp.cleanup()
        raise
    except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile) as exc:
        temp.cleanup()
        # Never expose signed URLs, credential-bearing requests or raw provider responses.
        raise DataPrepError("Không tải được dữ liệu. Kiểm tra kết nối, quyền truy cập và thử lại.") from exc

import io
import socket
import sqlite3
import zipfile

import pandas as pd
import pytest
import requests

from dataprep import remote
from dataprep.errors import DataPrepError
from dataprep.io import LoadOptions, sqlite_tables

KAGGLE = "https://www.kaggle.com/datasets/owner/sample"
DRIVE = "https://drive.google.com/file/d/abcdefghijk/view?usp=sharing"


class Response:
    def __init__(self, body=b"id,x\n001,3\n", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {"Content-Disposition": 'attachment; filename="data.csv"'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 5):
            yield self.body[i : i + 5]


@pytest.fixture
def network(monkeypatch):
    responses, calls = [], []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            calls.append((url, {**kwargs, "headers": dict(kwargs["headers"])}))
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(remote.requests, "Session", Session)
    monkeypatch.setattr(
        remote.socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 1, 6, "", ("8.8.8.8", 443))]
    )
    return responses, calls


def zip_bytes(files):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return data.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://www.kaggle.com/datasets/a/b",
        "https://evil.com/data.csv",
        "https://kaggle.com.evil.com/datasets/a/b",
        "https://user:pass@kaggle.com/datasets/a/b",
        "https://www.kaggle.com:8443/datasets/a/b",
        "https://www.kaggle.com/competitions/titanic",
        "https://drive.google.com/drive/folders/abcdefghijk",
        "https://docs.google.com/spreadsheets/d/abcdefghijk/edit",
    ],
)
def test_reject_unsupported_sources(url):
    with pytest.raises(DataPrepError):
        remote.parse_source(url)


def test_canonical_source_and_version():
    source = remote.parse_source(KAGGLE + "/versions/2?select=data.csv")
    assert source.download_url.endswith("?datasetVersionNumber=2")
    source = remote.parse_source(DRIVE + "&resourcekey=access-key")
    assert "resourcekey=access-key" in source.download_url
    assert "access-key" not in source.url
    assert remote.parse_source("https://drive.google.com/open?id=abcdefghijk").reference == source.reference


def test_download_zip_select_and_metadata(network):
    responses, calls = network
    responses.extend(
        [
            Response(
                status=302,
                headers={"Location": "https://storage.googleapis.com/bucket/data.zip?signature=private"},
            ),
            Response(
                zip_bytes(
                    {"nested/data.csv": "id,x\n001,3\n", "other.json": '[{"v": 2}]', "README.txt": "note"}
                ),
                headers={"Content-Disposition": 'attachment; filename="tables.zip"'},
            ),
        ]
    )
    result = remote.fetch_remote(KAGGLE, token="secret")
    try:
        assert result.files() == ["nested/data.csv", "other.json"]
        data = result.load("nested/data.csv")
        assert data.frame["id"].tolist() == ["001"]
        assert data.metadata["remote"] == {"provider": "Kaggle", "url": KAGGLE, "file": "nested/data.csv"}
        assert "secret" not in str(data.metadata)
        assert calls[0][1]["headers"]["Authorization"] == "Bearer secret"
        assert "Authorization" not in calls[1][1]["headers"]
        assert result.load("other.json").frame["v"].tolist() == [2]
        with pytest.raises(DataPrepError):
            result.materialize("missing.csv")
    finally:
        result.close()
    assert not result.path.exists()


def test_drive_confirmation(network):
    responses, calls = network
    html = b"""<html><form id="download-form" action="https://drive.usercontent.google.com/download" method="get"><input type="hidden" name="id" value="abcdefghijk"><input type="hidden" name="confirm" value="t"></form></html>"""
    responses.extend([Response(html, headers={"Content-Type": "text/html"}), Response()])
    result = remote.fetch_remote(DRIVE)
    try:
        assert result.load("data.csv").frame.shape == (1, 2)
        assert "confirm=t" in calls[1][0]
    finally:
        result.close()


@pytest.mark.parametrize(
    "location",
    [
        "http://storage.googleapis.com/a",
        "https://localhost/a",
        "https://evil.com/a",
        "https://drive.google.com/a",
    ],
)
def test_untrusted_redirect_rejected(network, location):
    responses, calls = network
    responses.append(Response(status=302, headers={"Location": location}))
    with pytest.raises(DataPrepError):
        remote.fetch_remote(KAGGLE, token="secret")
    assert len(calls) == 1


def test_private_dns_rejected(monkeypatch):
    monkeypatch.setattr(
        remote.socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 443))]
    )
    with pytest.raises(DataPrepError):
        remote._check_download_url("https://www.kaggle.com/api/v1/data", "Kaggle")


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_http_failures(network, status):
    network[0].append(Response(status=status))
    with pytest.raises(DataPrepError):
        remote.fetch_remote(KAGGLE)


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"Content-Length": "999999999"}, b"a"),
        ({"Content-Length": "15"}, b"a"),
        ({"Content-Type": "text/csv"}, b""),
        ({"Content-Type": "text/html"}, b"<html>Please sign in</html>"),
    ],
)
def test_bad_downloads(network, headers, body):
    network[0].append(Response(body, headers=headers))
    with pytest.raises(DataPrepError):
        remote.fetch_remote(DRIVE)


def test_stream_limit_without_header_and_error_redaction(network, monkeypatch):
    monkeypatch.setattr(remote, "MAX_BYTES", 8)
    network[0].append(Response(b"0123456789"))
    with pytest.raises(DataPrepError, match="giới hạn"):
        remote.fetch_remote(DRIVE)
    network[0].append(requests.ConnectionError("secret-token and signed-url"))
    with pytest.raises(DataPrepError) as caught:
        remote.fetch_remote(KAGGLE, token="secret-token")
    assert "secret-token" not in str(caught.value)


@pytest.mark.parametrize(
    "files",
    [
        {"../bad.csv": "a\n1"},
        {"C:/bad.csv": "a\n1"},
        {"/bad.csv": "a\n1"},
        {"readme.txt": "no data"},
    ],
)
def test_zip_validation(network, files):
    network[0].append(
        Response(zip_bytes(files), headers={"Content-Disposition": 'attachment; filename="data.zip"'})
    )
    with pytest.raises(DataPrepError):
        remote.fetch_remote(KAGGLE)


@pytest.mark.parametrize("has_name", [True, False])
def test_xlsx_is_not_treated_as_dataset_archive(network, has_name):
    data = io.BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_excel(data, index=False)
    network[0].append(
        Response(
            data.getvalue(),
            headers={"Content-Disposition": "attachment; filename*=UTF-8''bang%20diem.xlsx"}
            if has_name
            else {"Content-Type": "application/octet-stream"},
        )
    )
    result = remote.fetch_remote(DRIVE)
    try:
        assert result.files() == ["bang diem.xlsx" if has_name else "download.xlsx"]
        assert result.load(result.name).frame.shape == (2, 1)
    finally:
        result.close()


def test_sqlite_path_and_remote_selection(network, tmp_path):
    path = tmp_path / "data.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE scores (id TEXT, score REAL)")
        conn.execute("INSERT INTO scores VALUES ('001', 9)")
    assert sqlite_tables(path) == ["scores"]
    network[0].append(
        Response(path.read_bytes(), headers={"Content-Disposition": 'attachment; filename="data.sqlite"'})
    )
    result = remote.fetch_remote(DRIVE)
    try:
        assert sqlite_tables(result.materialize(result.name)) == ["scores"]
        assert result.load(result.name, LoadOptions(table="scores")).frame["id"].tolist() == ["001"]
    finally:
        result.close()


def test_legacy_credentials_and_unknown_filename(network):
    network[0].extend(
        [
            Response(status=302, headers={"Location": "https://storage.googleapis.com/a"}),
            Response(headers={"Content-Type": "application/octet-stream"}),
        ]
    )
    result = remote.fetch_remote(KAGGLE, username="name", key="key")
    try:
        assert network[1][0][1]["auth"] == ("name", "key")
        assert network[1][1][1]["auth"] is None
        assert result.load(result.name, filename="data.csv").frame.shape == (1, 2)
    finally:
        result.close()
    with pytest.raises(DataPrepError):
        remote.fetch_remote(KAGGLE, username="name")


def test_wrong_confirmation_file_and_host(network):
    network[0].append(
        Response(
            b'<html><form id="download-form" action="https://evil.com"><input type="hidden" name="id" value="another-file"></form></html>',
            headers={"Content-Type": "text/html"},
        )
    )
    with pytest.raises(DataPrepError):
        remote.fetch_remote(DRIVE)
    assert len(network[1]) == 1

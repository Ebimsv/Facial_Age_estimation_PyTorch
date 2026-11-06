"""Pinned Hugging Face downloads/config/TAR extraction with offline HTTP fixtures only."""

from __future__ import annotations

import hashlib
import io
import tarfile
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest
from omegaconf import OmegaConf
from PIL import Image

from scripts import download_data as download
from src.configs import load_download_config
from src.utils.download_config import huggingface_url, merge_download_config


def png(i):
    stream = io.BytesIO()
    Image.new("RGB", (4, 5), (i, 40, 80)).save(stream, format="PNG")
    return stream.getvalue()


def tar_bytes(parent="UTKFace", extras=()):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for i in range(3):
            data = png(i)
            member = tarfile.TarInfo(f"{parent + '/' if parent else ''}20_0_0_{i}.png")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for member, content in extras:
            archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, length=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    monkeypatch.setattr(download, "MIN_IMAGES", 3)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    monkeypatch.setattr(download, "load_kaggle_api", lambda: pytest.fail("No Kaggle on a Hugging Face path"))
    monkeypatch.setattr(download, "urlopen", lambda *a, **kw: pytest.fail("No real HTTP in unit tests"))


def config(payload, **hf):
    return load_download_config(updates={"huggingface": {
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), **hf}})


def fake_http(monkeypatch, payload, **kwargs):
    calls = []

    def open_url(request, *, timeout):
        calls.append(request)
        assert timeout == 30
        assert request.get_header("Authorization") is None
        return Response(payload, **kwargs)

    monkeypatch.setattr(download, "urlopen", open_url)
    return calls


def test_default_config_pins_public_mirror_and_is_immutable():
    cfg = load_download_config()
    assert cfg.source == "huggingface"
    assert cfg.huggingface.revision == "e80f21fb21fa3631380c3cb634b79be3ac2f4cba"
    assert cfg.huggingface.size_bytes == 106634631
    assert cfg.huggingface.sha256 == "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7"
    assert huggingface_url(cfg).endswith(f"/resolve/{cfg.huggingface.revision}/UTKFace.tar.gz")
    assert OmegaConf.is_readonly(cfg)


@pytest.mark.parametrize("patch,error", [
    ({"surprise": 1}, "surprise"), ({"source": "other"}, "source"), ({"dest": ""}, "dest"),
    ({"max_retries": -1}, "max_retries"), ({"max_retries": 6}, "max_retries"),
    ({"timeout_seconds": 0}, "timeout"), ({"timeout_seconds": float("nan")}, "timeout"),
    ({"huggingface": {"repo_id": "../bad"}}, "repo_id"),
    ({"huggingface": {"revision": "main"}}, "revision"),
    ({"huggingface": {"filename": "../file.tar.gz"}}, "filename"),
    ({"huggingface": {"sha256": "bad"}}, "sha256"),
    ({"huggingface": {"sha256": "???"}}, "sha256"),
    ({"huggingface": {"size_bytes": 0}}, "size_bytes"),
    ({"huggingface": {"size_bytes": 9 * 1024**3}}, "size_bytes"),
    ({"huggingface": {"image_subdir": "../escape"}}, "relative archive"),
    ({"kaggle": {"dataset_version": 0}}, "dataset_version"),
])
def test_invalid_download_settings_fail_before_destination_or_http(tmp_path, patch, error):
    with pytest.raises(ValueError, match=error):
        load_download_config(updates=patch)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("parent", ["UTKFace", "nested/UTKFace", ""])
def test_verified_tar_download_keeps_only_selected_images(tmp_path, monkeypatch, parent):
    member = tarfile.TarInfo("other/readme.txt")
    member.size = 4
    payload = tar_bytes(parent, [(member, b"keep")])
    cfg = config(payload, image_subdir=None)
    calls = fake_http(monkeypatch, payload)
    marker = tmp_path / "other.tar.gz"
    marker.write_bytes(b"user archive")
    target = download.download_from_huggingface(tmp_path, cfg)
    assert len(calls) == 1 and calls[0].full_url == huggingface_url(cfg)
    assert download.inspect_images(target) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.tar.gz", "utkcropped"]
    assert marker.read_bytes() == b"user archive"
    assert all(path.suffix == ".png" for path in target.iterdir())


@pytest.mark.parametrize("failure", ["checksum", "length", "partial", "oversize", "short_body"])
def test_bad_http_response_never_reaches_extraction(tmp_path, monkeypatch, failure):
    payload = tar_bytes()
    cfg = config(payload)
    if failure == "checksum":
        cfg = merge_download_config(cfg, {"huggingface": {"sha256": "0" * 64}})
    response_payload = payload + b"extra" if failure == "oversize" else payload
    if failure == "short_body":
        response_payload = payload[:-1]
    kwargs = {"length": len(payload) + 1} if failure == "length" else {"length": len(payload)}
    if failure == "partial":
        kwargs["status"] = 206
    fake_http(monkeypatch, response_payload, **kwargs)
    monkeypatch.setattr(download, "extract_tar_images", lambda *a: pytest.fail("Must verify bytes before extraction"))
    with pytest.raises(download.DownloadError):
        download.download_from_huggingface(tmp_path, cfg)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("code,attempts", [(403, 1), (404, 1), (429, 3), (503, 3)])
def test_http_failure_has_bounded_retries_and_no_kaggle_fallback(tmp_path, monkeypatch, code, attempts):
    payload = tar_bytes()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.invalid/?signed=PRIVATE", code, "private body", None, None)

    monkeypatch.setattr(download, "urlopen", fail)
    with pytest.raises(download.DownloadError) as error:
        download.download_from_huggingface(tmp_path, config(payload))
    assert len(calls) == attempts
    assert "PRIVATE" not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_retry_restarts_partial_owned_file_instead_of_appending(tmp_path, monkeypatch):
    payload = tar_bytes()
    calls = []

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise IncompleteRead(b"partial")
            return super().read(10)

    def interrupted(request, **kwargs):
        calls.append(1)
        return Interrupted(payload) if len(calls) == 1 else Response(payload)

    monkeypatch.setattr(download, "urlopen", interrupted)
    assert download.inspect_images(download.download_from_huggingface(tmp_path, config(payload))) == 3
    assert len(calls) == 2


def test_network_failure_is_clean_and_preserves_unrelated_files(tmp_path, monkeypatch):
    marker = tmp_path / "unrelated.zip"
    marker.write_bytes(b"keep")
    monkeypatch.setattr(download, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(URLError("PRIVATE DETAIL")))
    with pytest.raises(download.DownloadError) as error:
        download.download_from_huggingface(tmp_path, config(tar_bytes()))
    assert "PRIVATE" not in str(error.value)
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:/escape", "a\\escape", "a/../escape"])
def test_tar_traversal_is_rejected_before_publication(tmp_path, monkeypatch, name):
    member = tarfile.TarInfo(name)
    member.size = 3
    payload = tar_bytes(extras=[(member, b"bad")])
    fake_http(monkeypatch, payload)
    with pytest.raises(download.DownloadError, match="unsafe"):
        download.download_from_huggingface(tmp_path, config(payload))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_tar_links_and_special_files_are_never_extracted(tmp_path, monkeypatch, kind):
    member = tarfile.TarInfo("UTKFace/link.png")
    member.type = kind
    member.linkname = "UTKFace/20_0_0_0.png"
    payload = tar_bytes(extras=[(member, b"")])
    fake_http(monkeypatch, payload)
    with pytest.raises(download.DownloadError, match="link|special"):
        download.download_from_huggingface(tmp_path, config(payload))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["duplicate", "corrupt_tar", "decode", "size_limit", "members_limit"])
def test_tar_or_validation_failure_cleans_staging(tmp_path, monkeypatch, failure):
    member = tarfile.TarInfo("UTKFace/20_0_0_0.png")
    member.size = 3
    payload = tar_bytes(extras=[(member, b"dup")]) if failure == "duplicate" else tar_bytes()
    if failure == "corrupt_tar":
        payload = b"not a tar.gz"
    elif failure == "decode":
        # Replace selected image bytes, retaining valid name and a valid archive hash.
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for i in range(3):
                m = tarfile.TarInfo(f"UTKFace/20_0_0_{i}.png")
                m.size = 3
                archive.addfile(m, io.BytesIO(b"bad"))
        payload = stream.getvalue()
    elif failure == "size_limit":
        monkeypatch.setattr(download, "MAX_ARCHIVE_BYTES", 1)
    elif failure == "members_limit":
        monkeypatch.setattr(download, "MAX_MEMBERS", 1)
    fake_http(monkeypatch, payload)
    with pytest.raises(download.DownloadError):
        download.download_from_huggingface(tmp_path, config(payload))
    assert list(tmp_path.iterdir()) == []


def test_existing_target_is_not_replaced_or_downloaded(tmp_path):
    target = tmp_path / "utkcropped"
    target.mkdir()
    (target / "marker").write_text("keep")
    with pytest.raises(download.DownloadError, match="already exists"):
        download.download_from_huggingface(tmp_path, config(tar_bytes()))
    assert (target / "marker").read_text() == "keep"


def test_cli_defaults_to_huggingface_and_prints_pinned_preparation_provenance(tmp_path, monkeypatch, capsys):
    payload = tar_bytes()
    cfg_file = tmp_path / "download.yaml"
    cfg_file.write_text(OmegaConf.to_yaml(config(payload)))
    fake_http(monkeypatch, payload)
    monkeypatch.setattr("sys.argv", ["download_data", "--config", str(cfg_file), "--dest", str(tmp_path / "images")])
    download.main()
    output = capsys.readouterr().out
    assert "huggingface:py97/UTKFace-Cropped" in output and config(payload).huggingface.revision in output
    assert download.inspect_images(tmp_path / "images/utkcropped") == 3


def test_print_config_and_check_only_do_not_contact_either_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["download_data", "--print-config", "--dest", str(tmp_path / "absent")])
    download.main()
    assert huggingface_url(load_download_config()) in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["download_data", "--check-only", "--dest", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as error:
        download.main()
    assert error.value.code == 1 and not (tmp_path / "absent").exists()


def test_explicit_cli_provider_and_flags_override_download_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / "download.yaml"
    cfg_file.write_text("source: huggingface\nkaggle:\n  dataset_version: 4\n")
    calls = []
    monkeypatch.setattr(download, "download_from_kaggle", lambda dest, **kwargs: calls.append((dest, kwargs)) or dest / "utkcropped")
    monkeypatch.setattr("sys.argv", ["download_data", "--config", str(cfg_file), "--source", "kaggle",
                                     "--dataset-version", "7", "--image-subdir", "crop/UTKFace", "--dest", str(tmp_path / "out")])
    download.main()
    assert calls == [(tmp_path / "out", {"dataset_version": 7, "image_subdir": "crop/UTKFace"})]


def test_tar_extraction_keeps_increasing_offsets_for_gzip_performance(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for i in [2, 0, 1]:
            data = png(i)
            member = tarfile.TarInfo(f"UTKFace/20_0_0_{i}.png")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = stream.getvalue()
    fake_http(monkeypatch, payload)
    offsets = []
    original = tarfile.TarFile.extractfile

    def record(archive, member):
        offsets.append(member.offset_data)
        return original(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", record)
    download.download_from_huggingface(tmp_path, config(payload))
    assert len(offsets) == 3 and offsets == sorted(offsets)

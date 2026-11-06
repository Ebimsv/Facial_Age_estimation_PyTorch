"""Downloader guardrails, using fake APIs and tiny archives only (never Kaggle credentials)."""

from __future__ import annotations

import io
import stat
import subprocess
import zipfile

import pytest
from PIL import Image

from scripts import download_data as download


@pytest.fixture(autouse=True)
def small_dataset(monkeypatch):
    monkeypatch.setattr(download, "MIN_IMAGES", 3)


def png(value=0):
    stream = io.BytesIO()
    Image.new("RGB", (4, 5), (value, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def image_entries(parent="UTKFace"):
    return {f"{parent + '/' if parent else ''}20_0_0_{i}.png": png(i) for i in range(3)}


def archive_bytes(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


class FakeAPI:
    def __init__(self, entries=None, *, payload=None, archive_name=download.ARCHIVE_NAME):
        self.entries = image_entries() if entries is None else entries
        self.payload = payload
        self.archive_name = archive_name
        self.calls = []

    def authenticate(self):
        self.calls.append("auth")

    def dataset_download_files(self, reference, *, path, unzip, quiet):
        from pathlib import Path
        assert unzip is False and quiet is True
        self.calls.append(reference)
        (Path(path) / self.archive_name).write_bytes(
            archive_bytes(self.entries) if self.payload is None else self.payload)
        # An extra archive even in the private staging must never be selected.
        (Path(path) / "000-unrelated.zip").write_bytes(b"unrelated")


def local_images(parent):
    target = parent / "utkcropped"
    target.mkdir(parents=True)
    for name, content in image_entries("").items():
        (target / name).write_bytes(content)
    return target


@pytest.mark.parametrize("layout", ["UTKFace", "utkcropped", "nested/UTKFace", "nested/utkcropped/utkcropped", ""])
def test_download_discovers_supported_flat_and_nested_layouts(tmp_path, layout):
    unrelated = tmp_path / "other.zip"
    unrelated.write_bytes(b"user archive")
    api = FakeAPI(image_entries(layout))
    target = download.download_from_kaggle(tmp_path, dataset_version=7, api=api)
    assert api.calls == ["auth", "jangedoo/utkface-new/7"]
    assert download.inspect_images(target) == 3
    assert len(list(target.iterdir())) == 3
    assert unrelated.read_bytes() == b"user archive"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.zip", "utkcropped"]


def test_ambiguous_layout_requires_explicit_choice(tmp_path):
    entries = {**image_entries("UTKFace"), **image_entries("aligned/UTKFace")}
    with pytest.raises(download.DownloadError, match="ambiguous"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(entries))
    assert list(tmp_path.iterdir()) == []
    selected = download.download_from_kaggle(tmp_path, image_subdir="aligned/UTKFace", api=FakeAPI(entries))
    assert download.inspect_images(selected) == 3


def test_explicit_nonstandard_folder_and_flat_choice(tmp_path):
    with pytest.raises(download.DownloadError, match="layout"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(image_entries("cropped/crop_part1")))
    target = download.download_from_kaggle(tmp_path, image_subdir="cropped/crop_part1",
                                           api=FakeAPI(image_entries("cropped/crop_part1")))
    assert download.inspect_images(target) == 3


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "a/../b", "a//b", "a\\b", "C:/path", ""])
def test_invalid_image_subdir_fails_before_sdk_or_writes(tmp_path, bad):
    api = FakeAPI()
    with pytest.raises(ValueError, match="relative archive"):
        download.download_from_kaggle(tmp_path, image_subdir=bad, api=api)
    assert api.calls == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", [0, -1, True, 1.5])
def test_invalid_version_fails_before_sdk(tmp_path, version):
    api = FakeAPI()
    with pytest.raises(download.DownloadError, match="positive integer"):
        download.download_from_kaggle(tmp_path, dataset_version=version, api=api)
    assert not api.calls


def test_only_expected_archive_is_accepted(tmp_path):
    marker = tmp_path / "utkface-new.zip"
    marker.write_bytes(b"old unrelated file even with the expected name")
    with pytest.raises(download.DownloadError, match="Expected archive"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(archive_name="wrong.zip"))
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_bytes() == b"old unrelated file even with the expected name"


@pytest.mark.parametrize("payload", [b"not a ZIP", b"PK\x03\x04truncated"])
def test_malformed_archive_does_not_publish_or_touch_user_files(tmp_path, payload):
    marker = tmp_path / "notes.txt"
    marker.write_text("keep")
    with pytest.raises(download.DownloadError, match="Archive"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(payload=payload))
    assert list(tmp_path.iterdir()) == [marker] and marker.read_text() == "keep"


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt", "C:/escape.txt", "folder\\escape.txt", "a/../escape.txt"])
def test_unsafe_archive_members_are_rejected_before_extraction(tmp_path, name):
    with pytest.raises(download.DownloadError, match="unsafe"):
        download.download_from_kaggle(tmp_path, api=FakeAPI({**image_entries(), name: b"bad"}))
    assert list(tmp_path.iterdir()) == []


def test_archive_symlinks_and_duplicate_paths_are_rejected(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("UTKFace/link.png")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../escape")
    with pytest.raises(download.DownloadError, match="symlink"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(payload=stream.getvalue()))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("UTKFace/20_0_0_a.png", png())
        archive.writestr("UTKFace/20_0_0_a.png", png())
    with pytest.raises(download.DownloadError, match="duplicate"):
        download.download_from_kaggle(tmp_path, api=FakeAPI(payload=stream.getvalue()))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit,value", [("MAX_MEMBERS", 1), ("MAX_ARCHIVE_BYTES", 1), ("MAX_IMAGE_BYTES", 1)])
def test_archive_resource_limits_fail_before_publishing(tmp_path, monkeypatch, limit, value):
    monkeypatch.setattr(download, limit, value)
    with pytest.raises(download.DownloadError, match="limit"):
        download.download_from_kaggle(tmp_path, api=FakeAPI())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ["too_few", "bad_labels", "corrupt"])
def test_incomplete_or_corrupt_dataset_is_not_published(tmp_path, case):
    entries = image_entries()
    if case == "too_few":
        entries.pop(next(iter(entries)))
    elif case == "bad_labels":
        entries = {name.replace("20_0_0", "20_9_9"): content for name, content in entries.items()}
    else:
        entries[next(iter(entries))] = b"broken image"
    with pytest.raises(download.DownloadError):
        download.download_from_kaggle(tmp_path, api=FakeAPI(entries))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", ["auth", "network"])
def test_sdk_failures_are_sanitized_and_staging_cleaned(tmp_path, capsys, stage):
    api = FakeAPI()

    def fail(*args, **kwargs):
        print("DUMMY_PRIVATE_SDK_DETAIL")
        raise RuntimeError("DUMMY_PRIVATE_SDK_DETAIL")

    if stage == "auth":
        api.authenticate = fail
    else:
        api.dataset_download_files = fail
    with pytest.raises(download.DownloadError) as error:
        download.download_from_kaggle(tmp_path, api=api)
    assert "DUMMY_PRIVATE" not in str(error.value) + str(capsys.readouterr())
    assert list(tmp_path.iterdir()) == []


def test_missing_sdk_has_install_hint_without_installing_or_creating_output(tmp_path, monkeypatch):
    def missing(name):
        assert name == "kaggle"
        raise ModuleNotFoundError("kaggle")

    monkeypatch.setattr(download.importlib, "import_module", missing)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("No automatic installer may run"))
    with pytest.raises(download.DownloadError, match=r'\[data\]'):
        download.download_from_kaggle(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_sdk_import_auth_failure_is_sanitized(tmp_path, monkeypatch, capsys):
    def fail(name):
        print("DUMMY_IMPORT_SECRET")
        raise OSError("DUMMY_IMPORT_SECRET")

    monkeypatch.setattr(download.importlib, "import_module", fail)
    with pytest.raises(download.DownloadError, match="initialization") as error:
        download.download_from_kaggle(tmp_path)
    assert "DUMMY_IMPORT_SECRET" not in str(error.value) + str(capsys.readouterr())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("exists", [False, True])
def test_check_only_is_read_only_and_never_imports_sdk(tmp_path, monkeypatch, exists):
    if exists:
        local_images(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr("sys.argv", ["download_data", "--dest", str(tmp_path), "--check-only"])
    monkeypatch.setattr(download, "load_kaggle_api", lambda: pytest.fail("check-only must not import/authenticate"))
    monkeypatch.setattr(download.tempfile, "TemporaryDirectory", lambda **kw: pytest.fail("No staging during check-only"))
    if exists:
        download.main()
    else:
        with pytest.raises(SystemExit) as error:
            download.main()
        assert error.value.code == 1
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_existing_valid_dataset_skips_sdk_and_existing_invalid_target_is_preserved(tmp_path, monkeypatch):
    target = local_images(tmp_path)
    monkeypatch.setattr("sys.argv", ["download_data", "--dest", str(tmp_path)])
    monkeypatch.setattr(download, "load_kaggle_api", lambda: pytest.fail("Existing dataset must not authenticate"))
    download.main()
    next(target.iterdir()).write_bytes(b"now corrupt")
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(SystemExit) as error:
        download.main()
    assert error.value.code == 1
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_extraction_write_failure_cleans_only_staging(tmp_path, monkeypatch):
    marker = tmp_path / "user.zip"
    marker.write_bytes(b"keep")
    monkeypatch.setattr(download.shutil, "copyfileobj", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(download.DownloadError, match="extraction"):
        download.download_from_kaggle(tmp_path, api=FakeAPI())
    assert list(tmp_path.iterdir()) == [marker]


def test_publication_failure_rolls_back_own_files_and_preserves_unexpected_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.png").write_bytes(png())
    target = tmp_path / "published"

    def fail(stream, output, **kwargs):
        output.write(b"partial")
        (target / "user-note.txt").write_text("preserve")
        raise OSError("disk full")

    monkeypatch.setattr(download.shutil, "copyfileobj", fail)
    with pytest.raises(OSError):
        download.publish_images(source, target)
    assert sorted(p.name for p in target.iterdir()) == ["user-note.txt"]
    assert (target / "user-note.txt").read_text() == "preserve"
    assert (source / "a.png").read_bytes() == png()


def test_destination_created_during_download_is_not_replaced(tmp_path):
    api = FakeAPI()
    original = api.dataset_download_files

    def race(*args, **kwargs):
        original(*args, **kwargs)
        target = tmp_path / "utkcropped"
        target.mkdir()
        (target / "user-note.txt").write_text("keep")

    api.dataset_download_files = race
    with pytest.raises(download.DownloadError, match="already exists"):
        download.download_from_kaggle(tmp_path, api=api)
    assert (tmp_path / "utkcropped/user-note.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".utkface-download-*"))


def test_git_ignores_standard_credential_names_without_reading_credentials():
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=download.PROJECT_ROOT,
                            input="kaggle.json\nkaggle-backup.json\n.kaggle/access_token\naccess_token\n",
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["kaggle.json", "kaggle-backup.json", ".kaggle/access_token", "access_token"]

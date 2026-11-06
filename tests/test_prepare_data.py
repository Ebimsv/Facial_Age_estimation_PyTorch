"""Offline data protocol tests: rejected inputs, repeatability, leakage and tampering."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
from PIL import Image

from scripts import prepare_data
from src.data.dataset import build_dataloaders
from src.data.validation import COLUMNS, describe_splits, validate_partition, validate_table


def make_images(root, ages=(20, 30), per_age=40):
    root.mkdir()
    for i, age in enumerate(age for age in ages for _ in range(per_age)):
        Image.new("RGB", (6, 7), (i % 256, i // 256, 100)).save(root / f"{age}_0_0_{i:05d}.png")
    return root


def prepare(monkeypatch, root, output, *extra):
    monkeypatch.setattr("sys.argv", ["prepare_data", "--dataset-dir", str(root), "--csv-dir", str(output),
                                     "--source", "synthetic", "--dataset-version", "test-v1", *extra])
    prepare_data.main()
    return json.loads((output / "split_manifest.json").read_text())


def frames(output):
    return {name: pd.read_csv(output / filename) for name, filename in
            {"train": "train_set.csv", "val": "valid_set.csv", "test": "test_set.csv"}.items()}


def test_protocol_is_repeatable_and_preserves_complete_partition(tmp_path, monkeypatch):
    root = make_images(tmp_path / "images")
    first, second = tmp_path / "v2-a", tmp_path / "v2-b"
    one = prepare(monkeypatch, root, first, "--check-decode")
    two = prepare(monkeypatch, root, second, "--check-decode")
    assert one == two  # no wall-clock timestamp or output directory in semantic manifest
    assert one["source"]["version"] == "test-v1"
    assert one["validation"]["identity_leakage"] == "unknown"
    assert one["validation"]["decode_checked"] is True
    assert one["counts"] == {"master": 80, "train": 64, "val": 8, "test": 8, "scanned": 80, "rejected": 0}
    for filename in one["csv_files"]:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    splits = frames(first)
    validate_partition(splits, pd.read_csv(first / "utkface_dataset.csv"))
    for frame in splits.values():
        assert set(frame.age) == {20, 30}
    metadata = describe_splits(first, root, splits, (0, 100))
    assert metadata["content_verified"] and metadata["master_verified"]


def test_reject_report_covers_bad_labels_file_types_decode_and_duplicates(tmp_path, monkeypatch):
    root = make_images(tmp_path / "images")
    (root / "20_0_0_directory.png").mkdir()
    (root / "20_0_0_notes.txt").write_text("not an image")
    (root / "20_0_0_broken.png").write_bytes(b"broken png")
    for name in ["20_9_0_bad.png", "20_0_9_bad.png", "-1_0_0_bad.png", "20_0_0.png", "bad.png", "85_0_0_old.png"]:
        Image.new("RGB", (2, 2)).save(root / name)
    # All members of a duplicate group are rejected, including conflicting ages.
    duplicate = sorted(root.glob("20_0_0_0*.png"))[0]
    shutil.copyfile(duplicate, root / "30_0_0_copy.png")
    out = tmp_path / "v2"
    manifest = prepare(monkeypatch, root, out, "--check-decode")
    rejected = pd.read_csv(out / "rejected_samples.csv").set_index("image_name").reason.to_dict()
    assert rejected["20_0_0_broken.png"] == "decode_failed"
    assert rejected["85_0_0_old.png"] == "age_filter"
    assert rejected["20_9_0_bad.png"] == "invalid_filename_or_labels"
    assert rejected[duplicate.name] == rejected["30_0_0_copy.png"] == "duplicate_content"
    assert "20_0_0_directory.png" in rejected and "20_0_0_notes.txt" in rejected
    assert len(manifest["duplicate_groups"]) == 1
    assert manifest["counts"]["scanned"] == 90
    assert manifest["counts"]["master"] + manifest["counts"]["rejected"] == 90


def test_hash_and_decode_checks_are_independent_opt_out_is_recorded(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    for i in range(40):
        (root / f"20_0_0_{i}.jpg.chip.jpg").write_bytes(b"not decoded")
    manifest = prepare(monkeypatch, root, tmp_path / "out", "--skip-content-hash")
    assert manifest["content_sha256"] is None
    assert manifest["validation"]["decode_checked"] is False
    assert manifest["validation"]["duplicate_policy"] == "not_checked"
    assert not describe_splits(tmp_path / "out", root, frames(tmp_path / "out"))["content_verified"]


@pytest.mark.parametrize("kind", ["csv", "image", "missing_hash", "schema", "bool_schema", "hash_flag", "duplicate_hash"])
def test_manifest_rejects_changed_csv_images_or_incomplete_metadata(tmp_path, monkeypatch, kind):
    root = make_images(tmp_path / "images")
    out = tmp_path / "v2"
    manifest = prepare(monkeypatch, root, out)
    if kind == "csv":
        csv = pd.read_csv(out / "train_set.csv")
        csv.loc[0, "age"] += 1
        csv.to_csv(out / "train_set.csv", index=False)
    elif kind == "image":
        next(root.iterdir()).write_bytes(b"changed")
    else:
        if kind == "missing_hash":
            manifest["content_sha256"].pop(next(iter(manifest["content_sha256"])))
        elif kind == "schema":
            manifest["schema_version"] = 999
        elif kind == "bool_schema":
            manifest["schema_version"] = True
        elif kind == "hash_flag":
            manifest["validation"]["content_hashed"] = False
        else:
            names = list(manifest["content_sha256"])
            manifest["content_sha256"][names[1]] = manifest["content_sha256"][names[0]]
        (out / "split_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_dataloaders(root, out, None, None, None, batch_size=2, num_workers=0)


def test_existing_output_and_legacy_csvs_are_never_overwritten(tmp_path, monkeypatch):
    root = make_images(tmp_path / "images")
    out = tmp_path / "legacy"
    out.mkdir()
    marker = out / "train_set.csv"
    marker.write_bytes(b"historical protocol")
    with pytest.raises(FileExistsError, match="new --csv-dir"):
        prepare(monkeypatch, root, out)
    assert marker.read_bytes() == b"historical protocol"
    assert sorted(p.name for p in out.iterdir()) == ["train_set.csv"]


@pytest.mark.parametrize("counts", [[1, 40], [2, 40], [4, 4]])
def test_sparse_age_strata_fail_before_any_output(tmp_path, monkeypatch, counts):
    root = tmp_path / "images"
    root.mkdir()
    for age, count in zip([20, 30], counts):
        for i in range(count):
            Image.new("RGB", (3, 3), (age, i, 0)).save(root / f"{age}_0_0_{i}.png")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="infeasible"):
        prepare(monkeypatch, root, out)
    assert not out.exists()


def test_empty_scan_and_no_silent_unstratified_fallback(tmp_path):
    with pytest.raises(ValueError, match="No valid images"):
        prepare_data.scan_images(tmp_path, 85)
    with pytest.raises(ValueError, match="empty"):
        prepare_data.stratified_split(pd.DataFrame(columns=COLUMNS), .8, .1, 42)
    tiny_test = pd.DataFrame({"image_name": [f"{i}.png" for i in range(20)], "age": [20] * 10 + [30] * 10,
                             "gender": "Male", "ethnicity": "White"})
    with pytest.raises(ValueError, match="held_out=1"):
        prepare_data.stratified_split(tiny_test, .75, .2499999, 42)


def test_split_is_independent_of_dataframe_order(tmp_path):
    root = make_images(tmp_path / "images")
    table, _, _, _ = prepare_data.scan_images(root, 85)
    one = prepare_data.stratified_split(table, .8, .1, 42)
    two = prepare_data.stratified_split(table.sample(frac=1, random_state=10), .8, .1, 42)
    for first, second in zip(one, two):
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("mutation,error", [
    ({"age": "bad"}, "age targets"), ({"age": -1}, "age targets"),
    ({"age": float("inf")}, "age targets"), ({"age": 101}, "age targets"),
    ({"gender": "Unknown"}, "gender"), ({"ethnicity": "unknown"}, "ethnicity"),
    ({"image_name": "../outside.png"}, "inside root"), ({"image_name": "/outside.png"}, "inside root"),
    ({"image_name": "nested/../image.png"}, "inside root"), ({"image_name": "./image.png"}, "inside root"),
    ({"image_name": "image.txt"}, "extension"), ({"image_name": "absent.png"}, "regular file"),
    ({"image_name": None}, "image_name"), ({"image_name": "folder\\image.png"}, "image_name"),
])
def test_csv_contract_rejects_bad_values_and_paths(tmp_path, mutation, error):
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    row = {"image_name": "image.png", "age": 20, "gender": "Male", "ethnicity": "White", **mutation}
    with pytest.raises(ValueError, match=error):
        validate_table(pd.DataFrame([row]), root=tmp_path, age_range=(0, 100))


def test_symlink_escape_and_duplicate_resolved_paths_are_rejected(tmp_path):
    root = make_images(tmp_path / "images", per_age=3)
    (root / "escape.png").symlink_to(tmp_path / "outside.png")
    row = {"image_name": "escape.png", "age": 20, "gender": "Male", "ethnicity": "White"}
    with pytest.raises(ValueError, match="escapes root"):
        validate_table(pd.DataFrame([row]), root=root)
    original = sorted(root.glob("20*.png"))[0]
    (root / "alias.png").symlink_to(original)
    a = pd.DataFrame([{**row, "image_name": original.name}])
    b = pd.DataFrame([{**row, "image_name": "alias.png"}])
    with pytest.raises(ValueError, match="overlap"):
        validate_partition({"train": a, "test": b}, root=root)


def test_duplicate_rows_overlap_and_mismatched_master_labels_are_rejected():
    a = pd.DataFrame([{"image_name": "a.png", "age": 20, "gender": "Male", "ethnicity": "White"}])
    with pytest.raises(ValueError, match="duplicate image_name"):
        validate_table(pd.concat([a, a]))
    with pytest.raises(ValueError, match="overlap"):
        validate_partition({"train": a, "test": a})
    b = a.assign(image_name="b.png")
    with pytest.raises(ValueError, match="union/labels"):
        validate_partition({"train": a, "test": b}, pd.concat([a, b.assign(age=21)]))


def test_decimal_split_sizes_do_not_gain_an_image_from_binary_rounding(tmp_path):
    root = make_images(tmp_path / "images", ages=(20,), per_age=40)
    table, _, _, _ = prepare_data.scan_images(root, 85)
    train, val, test = prepare_data.stratified_split(table, .6, .25, 42)
    assert [len(train), len(val), len(test)] == [24, 10, 6]


def test_master_helper_and_symlink_alias_split_do_not_overwrite(tmp_path, monkeypatch):
    root = make_images(tmp_path / "images")
    master = tmp_path / "master.csv"
    master.write_text("legacy")
    with pytest.raises(FileExistsError):
        prepare_data.create_master_csv(root, master, 85)
    assert master.read_text() == "legacy"
    (root / "20_0_0_alias.png").symlink_to(sorted(root.glob("20*.png"))[0])
    out = tmp_path / "v2"
    with pytest.raises(ValueError, match="overlap"):
        prepare(monkeypatch, root, out, "--skip-content-hash")
    assert not out.exists()


def test_committed_legacy_csv_partition_remains_valid_without_images():
    from pathlib import Path
    root = Path(__file__).parents[1] / "csv_dataset"
    splits = {name: validate_table(frame) for name, frame in frames(root).items()}
    master = validate_table(pd.read_csv(root / "utkface_dataset.csv"))
    validate_partition(splits, master)
    assert {name: len(frame) for name, frame in splits.items()} == {"train": 15822, "val": 2793, "test": 4654}

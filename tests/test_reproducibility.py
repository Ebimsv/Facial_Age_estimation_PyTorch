"""Repeatability within the same runtime/worker configuration, not cross-device equivalence."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.configs import load_config
from src.data.dataset import build_dataloaders
from src.utils.environment import collect_environment
from src.utils.seed import seed_worker, set_seed


class RandomTransform:
    def __call__(self, image):
        return torch.tensor([random.random(), float(np.random.random()), torch.rand(()).item()])


def loader_data(tmp_path):
    root, csvs = tmp_path / "images", tmp_path / "csvs"
    root.mkdir()
    csvs.mkdir()
    for offset, split in enumerate(["train", "valid", "test"]):
        rows = []
        for i in range(8):
            name = f"{split}_{i}.png"
            Image.new("RGB", (3, 3), (offset, i, 0)).save(root / name)
            rows.append({"image_name": name, "age": i + 20, "gender": "Male", "ethnicity": "White"})
        pd.DataFrame(rows).to_csv(csvs / f"{split}_set.csv", index=False)
    return root, csvs


@pytest.mark.parametrize("workers", [0, 2])
def test_repeated_loaders_have_same_shuffle_and_all_rng_streams(tmp_path, workers):
    root, csvs = loader_data(tmp_path)

    def collect():
        set_seed(17)
        loaders = build_dataloaders(root, csvs, RandomTransform(), RandomTransform(), RandomTransform(),
                                    batch_size=2, num_workers=workers, seed=17)
        # Validation iteration cannot consume the training generator.
        list(loaders[1])
        return [(images, ages) for images, ages, _, _ in loaders[0]]

    first, second = collect(), collect()
    for (images, ages), (other_images, other_ages) in zip(first, second):
        torch.testing.assert_close(images, other_images, rtol=0, atol=0)
        torch.testing.assert_close(ages, other_ages, rtol=0, atol=0)
    if workers:
        assert torch.unique(torch.cat([batch[0] for batch in first]), dim=0).shape[0] == 8


def test_loader_shuffle_is_independent_of_global_torch_rng_and_val_iteration(tmp_path):
    root, csvs = loader_data(tmp_path)
    a = build_dataloaders(root, csvs, RandomTransform(), None, None, batch_size=2, num_workers=0, seed=7)
    b = build_dataloaders(root, csvs, RandomTransform(), None, None, batch_size=2, num_workers=0, seed=7)
    torch.rand(200)
    list(DataLoader(a[1].dataset, batch_size=2, collate_fn=lambda batch: batch))
    order_a = torch.cat([ages for _, ages, _, _ in a[0]])
    order_b = torch.cat([ages for _, ages, _, _ in b[0]])
    torch.testing.assert_close(order_a, order_b, rtol=0, atol=0)
    assert a[0].data_provenance["protocol"] == "unversioned_csv"
    assert a[0].data_provenance["content_verified"] is False


def test_worker_seeder_and_transform_support_spawn(tmp_path):
    root, csvs = loader_data(tmp_path)
    source = build_dataloaders(root, csvs, RandomTransform(), None, None, batch_size=2, num_workers=0)[0].dataset

    def collect():
        loader = DataLoader(source, batch_size=2, num_workers=1, multiprocessing_context="spawn",
                            generator=torch.Generator().manual_seed(4), worker_init_fn=seed_worker)
        return torch.cat([images for images, _, _, _ in loader])

    torch.testing.assert_close(collect(), collect(), rtol=0, atol=0)


def test_strict_determinism_is_explicit_and_restores_process_flags(monkeypatch):
    before = torch.are_deterministic_algorithms_enabled()
    deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "temporary")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setenv("PYTHONHASHSEED", "original")
    try:
        set_seed(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert os.environ["PYTHONHASHSEED"] == "original"
        set_seed(42, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert load_config().training.deterministic is False
        assert load_config(overrides=["training.deterministic=true"]).training.deterministic is True
    finally:
        torch.use_deterministic_algorithms(before)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark


def test_environment_is_plain_and_does_not_dump_environment_variables(monkeypatch):
    import json
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    metadata = collect_environment()
    assert isinstance(metadata["packages"]["torch"], str)
    assert "must-not-appear" not in json.dumps(metadata)
    assert isinstance(metadata["python"], str)

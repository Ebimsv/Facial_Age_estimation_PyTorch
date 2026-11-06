"""Weights-only-safe RNG and epoch-boundary DataLoader state."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from src.utils.config import DISTRIBUTION_LOSSES, to_plain_config


def capture_rng(*, cuda: bool = False) -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": {"name": str(state[0]), "keys": torch.tensor(state[1].astype(np.int64)),
                  "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
        "cuda": torch.cuda.get_rng_state_all() if cuda else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state["name"], np_state["keys"].cpu().numpy().astype(np.uint32),
                         np_state["position"], np_state["has_gauss"], np_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _name(value) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def capture_loaders(loaders: dict) -> dict:
    """Capture standard nonpersistent loaders, after their epoch iterators are exhausted."""
    result, generators = {}, {}
    for name, loader in loaders.items():
        if loader.persistent_workers or type(loader.sampler) not in {RandomSampler, SequentialSampler} or type(loader.batch_sampler) is not BatchSampler:
            raise ValueError("Epoch-boundary resume supports standard samplers/batching and persistent_workers=False")
        context = loader.multiprocessing_context
        spec = {"dataset_type": _name(loader.dataset), "samples": len(loader.dataset), "batches": len(loader),
                "batch_size": loader.batch_size, "drop_last": loader.drop_last, "num_workers": loader.num_workers,
                "sampler": _name(loader.sampler), "replacement": getattr(loader.sampler, "replacement", False),
                "num_samples": getattr(loader.sampler, "num_samples", len(loader.dataset)),
                "prefetch_factor": loader.prefetch_factor,
                "in_order": getattr(loader, "in_order", True),
                "context": context.get_start_method() if context is not None else None}
        entry = {"spec": spec}
        for key, generator in {"generator": loader.generator, "sampler_generator": getattr(loader.sampler, "generator", None)}.items():
            entry[key] = generator.get_state() if generator is not None else None
            if generator is not None:
                generators[f"{name}.{key}"] = generator
        result[name] = entry
    result["aliases"] = [(a, b) for a in generators for b in generators if a < b and generators[a] is generators[b]]
    return result


def validate_loaders(loaders: dict, saved: dict) -> None:
    current = capture_loaders(loaders)
    if current["aliases"] != saved["aliases"]:
        raise ValueError("Resume loader generator sharing changed")
    for name in loaders:
        if current[name]["spec"] != saved[name]["spec"]:
            raise ValueError(f"Resume {name} loader contract changed (batch/worker/sampler/dataset)")
        for key in ("generator", "sampler_generator"):
            if (current[name][key] is None) != (saved[name][key] is None):
                raise ValueError(f"Resume {name} {key} configuration changed")


def restore_loaders(loaders: dict, state: dict) -> None:
    for name, loader in loaders.items():
        for key, generator in {"generator": loader.generator, "sampler_generator": getattr(loader.sampler, "generator", None)}.items():
            if generator is not None:
                generator.set_state(state[name][key].cpu())


def training_config_signature(cfg) -> dict:
    """Only locations/logging may change on resume; the OneCycle horizon is immutable."""
    result = to_plain_config(cfg)
    result.pop("logging")
    result.pop("device")
    for key in ("root_dir", "csv_dir"):
        result["data"].pop(key)
    if result["loss"]["type"] in DISTRIBUTION_LOSSES:
        result["loss"]["type"] = "l1_mean_variance"
    return result

"""UTKFace dataset and dataloader builder.

This module is **side-effect free**: importing it never touches the filesystem,
trains a model, or allocates GPU memory.  All I/O happens inside the classes
or inside ``build_dataloaders``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from src.data.validation import describe_splits, image_path, validate_table
from src.utils.seed import seed_worker


class UTKDataset(Dataset):
    """PyTorch Dataset for the UTKFace cropped dataset.

    Each image filename encodes ``<age>_<gender>_<ethnicity>_<timestamp>.jpg``.
    The CSV produced by ``scripts/prepare_data.py`` stores these labels so we
    never rely on filename parsing at training time.

    Args:
        root_dir:  Path to the folder containing UTKFace images.
        csv_file:  Path to the CSV with columns [image_name, age, gender, ethnicity].
        transform: torchvision transform pipeline applied to each image.
        age_range: Optional inclusive target support, checked before image I/O.
    """

    GENDER_MAP = {0: "Male", 1: "Female"}
    ETHNICITY_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

    def __init__(
        self,
        root_dir: str | Path,
        csv_file: str | Path,
        transform: Compose | None = None,
        age_range: tuple[int, int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.data = validate_table(pd.read_csv(csv_file), context=str(csv_file),
                                   root=self.root_dir, age_range=age_range)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.data.iloc[idx]
        img_path = image_path(self.root_dir, row["image_name"])
        try:
            with Image.open(img_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Could not decode dataset image: {img_path}") from exc
        if self.transform is not None:
            image = self.transform(image)

        age = torch.tensor([row["age"]], dtype=torch.float32)
        gender = str(row["gender"])
        ethnicity = str(row["ethnicity"])

        return image, age, gender, ethnicity

    def get_age_array(self) -> list[int]:
        """Return all ages as a list (used for stratified sampling)."""
        return self.data["age"].tolist()


def build_dataloaders(
    root_dir: str | Path,
    csv_dir: str | Path,
    train_transform: Compose,
    val_transform: Compose,
    test_transform: Compose,
    batch_size: int = 96,
    eval_batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    age_range: tuple[int, int] | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train / val / test DataLoaders from pre-split CSVs.

    Args:
        root_dir:        Folder containing the UTKFace images.
        csv_dir:         Folder containing train_set.csv, valid_set.csv, test_set.csv.
        train_transform: Augmented transform for training.
        val_transform:   Deterministic transform for validation.
        test_transform:  Deterministic transform for testing.
        batch_size:      Batch size for training.
        eval_batch_size: Batch size for validation/testing.
        num_workers:     DataLoader worker processes.
        pin_memory:      Pin memory for GPU transfers (recommended when using CUDA).
        age_range:       Optional inclusive target support for every split CSV.
        seed:            Independent loader generators; custom Python/NumPy worker RNG seeds.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    csv_dir = Path(csv_dir)

    train_ds = UTKDataset(root_dir, csv_dir / "train_set.csv", train_transform, age_range)
    val_ds = UTKDataset(root_dir, csv_dir / "valid_set.csv", val_transform, age_range)
    test_ds = UTKDataset(root_dir, csv_dir / "test_set.csv", test_transform, age_range)
    for name, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if len(dataset) == 0:
            raise ValueError(f"{name} split is empty; prepare nonempty split CSVs before training")
    if len(train_ds) < batch_size:
        raise ValueError("Training loader would have zero batches with drop_last=True; reduce training.batch_size")
    provenance = describe_splits(csv_dir, Path(root_dir),
                                 {"train": train_ds.data, "val": val_ds.data, "test": test_ds.data}, age_range)
    provenance["loader_seed"] = seed

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # keeps batch sizes uniform for OneCycleLR
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + 1),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + 2),
        worker_init_fn=seed_worker,
    )

    for loader in (train_loader, val_loader, test_loader):
        loader.data_provenance = provenance

    return train_loader, val_loader, test_loader

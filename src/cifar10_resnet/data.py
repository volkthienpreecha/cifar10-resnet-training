"""CIFAR-10 transforms and deterministic data-loader construction."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def split_indices(
    dataset_size: int, train_size: int, validation_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Return deterministic, disjoint training and validation indices."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if train_size <= 0 or validation_size <= 0:
        raise ValueError("split sizes must be positive")
    if train_size + validation_size > dataset_size:
        raise ValueError("requested split is larger than the dataset")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(dataset_size, generator=generator).tolist()
    return order[:train_size], order[train_size : train_size + validation_size]


def cifar10_transforms(train: bool) -> Callable[[Any], torch.Tensor]:
    """Build the notebook's augmentation or validation transform."""

    from torchvision import transforms

    operations: list[Callable[..., Any]] = []
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if train:
        operations.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        )
    return transforms.Compose(operations)


class _TransformSubset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, dataset: Dataset[Any], indices: list[int], transform: Callable[..., Any]):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[self.indices[index]]
        return self.transform(image), int(target)


def build_cifar10_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Download CIFAR-10 and return deterministic 40k/10k loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=data_dir, train=True, download=True)
    train_indices, validation_indices = split_indices(len(dataset), 40_000, 10_000, seed)
    train_dataset = _TransformSubset(dataset, train_indices, cifar10_transforms(train=True))
    validation_dataset = _TransformSubset(
        dataset, validation_indices, cifar10_transforms(train=False)
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader

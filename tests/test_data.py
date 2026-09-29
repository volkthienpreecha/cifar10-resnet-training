from pathlib import Path

import pytest

from cifar10_resnet.data import build_cifar10_loaders, split_indices


def test_split_indices_are_deterministic_and_disjoint() -> None:
    first_train, first_validation = split_indices(100, 60, 20, seed=7)
    second_train, second_validation = split_indices(100, 60, 20, seed=7)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 60
    assert len(first_validation) == 20
    assert set(first_train).isdisjoint(first_validation)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_loader_builder_rejects_invalid_batch_size_before_download(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build_cifar10_loaders(Path("unused"), batch_size=batch_size, seed=1)


def test_split_indices_reject_oversized_request() -> None:
    with pytest.raises(ValueError, match="dataset"):
        split_indices(10, 8, 3, seed=1)

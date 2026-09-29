import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10_resnet.engine import accuracy, evaluate, resolve_device, train_epoch


def test_accuracy_counts_argmax_predictions() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    targets = torch.tensor([1, 0, 0])
    assert accuracy(logits, targets) == pytest.approx(2 / 3)


def test_train_and_evaluate_return_bounded_metrics() -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])),
        batch_size=4,
    )
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    training = train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    validation = evaluate(model, loader, criterion, torch.device("cpu"))

    for metrics in (training, validation):
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1


def test_evaluate_rejects_empty_loader() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        evaluate(nn.Linear(4, 2), loader, nn.CrossEntropyLoss(), torch.device("cpu"))


def test_resolve_device_accepts_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_uses_mps_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")

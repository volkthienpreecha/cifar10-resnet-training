"""Device-aware training and evaluation helpers."""

from collections.abc import Iterable

import torch
from torch import nn


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, Apple Metal, or CPU in that order."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(normalized)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy for one batch."""

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have shapes (N, C) and (N,)")
    if targets.numel() == 0:
        raise ValueError("cannot compute accuracy for an empty batch")
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.to(device)
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train ``model`` for one epoch and return mean loss and accuracy."""

    return _run_epoch(model, loader, criterion, device, optimizer)


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate ``model`` without gradient tracking."""

    return _run_epoch(model, loader, criterion, device, optimizer=None)

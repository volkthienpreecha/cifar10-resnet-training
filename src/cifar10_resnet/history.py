"""Parsing and presentation helpers for historical notebook training logs."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    loss: float
    train_accuracy: float
    val_accuracy: float


_EPOCH_PATTERN = re.compile(
    r"^epoch\s+(?P<epoch>\d+)\s+"
    r"loss:\s*(?P<loss>\d+(?:\.\d+)?)\s+"
    r"time:\s*\d+(?:\.\d+)?\s+"
    r"train acc:\s*(?P<train>\d+(?:\.\d+)?)\s+"
    r"val acc:\s*(?P<validation>\d+(?:\.\d+)?)\s*$",
    flags=re.MULTILINE,
)


def parse_epoch_history(text: str) -> list[EpochMetrics]:
    """Parse complete epoch records while ignoring unrelated notebook output."""

    return [
        EpochMetrics(
            epoch=int(match.group("epoch")),
            loss=float(match.group("loss")),
            train_accuracy=float(match.group("train")),
            val_accuracy=float(match.group("validation")),
        )
        for match in _EPOCH_PATTERN.finditer(text)
    ]

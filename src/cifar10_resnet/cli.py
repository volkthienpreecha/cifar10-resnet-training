"""Command-line training entry point."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import build_cifar10_loaders
from .engine import evaluate, resolve_device, train_epoch
from .models import FCNN, ResNet18, SimpleCNN, TunedCNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fcnn", "cnn", "resnet18", "tuned"], default="resnet18")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def _build_model(name: str) -> nn.Module:
    factories = {
        "fcnn": lambda: FCNN([3 * 32 * 32, 100, 10]),
        "cnn": SimpleCNN,
        "resnet18": ResNet18,
        "tuned": TunedCNN,
    }
    return factories[name]()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, validation_loader = build_cifar10_loaders(
        args.data_dir, args.batch_size, args.seed, args.num_workers
    )
    model = _build_model(args.model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if args.model == "tuned" else 0.0)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1 if args.model == "tuned" else 0.001, momentum=0.9
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(json.dumps(history[-1]))
    torch.save(model.state_dict(), args.output_dir / "model.pth")
    (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()

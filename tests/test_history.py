from cifar10_resnet.history import EpochMetrics, parse_epoch_history


def test_parser_extracts_complete_epoch_metrics() -> None:
    text = "epoch 120\tloss: 270.073\ttime: 17.871\ttrain acc: 0.9184\tval acc: 0.8860"
    assert parse_epoch_history(text) == [
        EpochMetrics(epoch=120, loss=270.073, train_accuracy=0.9184, val_accuracy=0.8860)
    ]


def test_parser_preserves_multiple_decimal_values_in_order() -> None:
    text = (
        "epoch 1 loss: 1.2345 time: 2.0 train acc: 0.1250 val acc: 0.1001\n"
        "epoch 2 loss: 0.9999 time: 2.0 train acc: 0.2501 val acc: 0.2002"
    )
    parsed = parse_epoch_history(text)
    assert [row.epoch for row in parsed] == [1, 2]
    assert parsed[1].val_accuracy == 0.2002


def test_parser_ignores_progress_text_and_partial_lines() -> None:
    text = "50%|#####| progress\nepoch 1 loss: 0.5\ntraining complete"
    assert parse_epoch_history(text) == []


def test_parser_returns_empty_list_when_no_metrics_exist() -> None:
    assert parse_epoch_history("no recorded metrics") == []

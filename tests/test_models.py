import pytest
import torch
from torch import nn

from cifar10_resnet.models import (
    FCNN,
    ResidualBlock,
    ResNet18,
    SimpleCNN,
    TunedCNN,
    count_trainable_parameters,
)


@pytest.mark.parametrize(
    "model",
    [FCNN([3 * 32 * 32, 100, 10]), SimpleCNN(), ResNet18(), TunedCNN()],
)
def test_classifier_returns_one_logit_vector_per_image(model: nn.Module) -> None:
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_projection_shortcut_matches_downsampled_shape() -> None:
    block = ResidualBlock(64, 128, stride=2)
    output = block(torch.randn(2, 64, 32, 32))
    assert output.shape == (2, 128, 16, 16)
    assert not isinstance(block.shortcut, nn.Identity)


def test_equal_shape_residual_block_uses_identity_shortcut() -> None:
    assert isinstance(ResidualBlock(64, 64).shortcut, nn.Identity)


def test_zero_residual_branch_leaves_relu_shortcut() -> None:
    block = ResidualBlock(4, 4).eval()
    for parameter in (*block.conv1.parameters(), *block.bn1.parameters(), *block.conv2.parameters(), *block.bn2.parameters()):
        nn.init.zeros_(parameter)
    inputs = torch.randn(2, 4, 8, 8)
    assert torch.allclose(block(inputs), torch.relu(block.shortcut(inputs)))


@pytest.mark.parametrize("layer_dims", [[], [10], [10, 0], [10, -4, 2]])
def test_fcnn_rejects_invalid_layer_dimensions(layer_dims: list[int]) -> None:
    with pytest.raises(ValueError):
        FCNN(layer_dims)


def test_residual_block_rejects_nonpositive_stride() -> None:
    with pytest.raises(ValueError):
        ResidualBlock(16, 16, stride=0)


def test_parameter_count_reports_trainable_parameters() -> None:
    assert count_trainable_parameters(SimpleCNN()) > 0

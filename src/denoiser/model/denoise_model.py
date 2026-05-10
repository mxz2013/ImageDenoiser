"""PyTorch implementation and weight loading utilities for the denoiser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Two-layer residual block used throughout the denoising network. Input and output have the same shape."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Args:
            channels: Number of input and output feature channels.
        """
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        return x + self.res(x)


class PriorNet(nn.Module):
    """U-Net-like denoising prior network matching the supplied diagram."""

    def __init__(self) -> None:
        """Initialize the architecture with names matching ``weights.csv``."""
        super().__init__()
        self.m_head = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)

        self.m_down1 = self._make_down_stage(32, 64)
        self.m_down2 = self._make_down_stage(64, 128)
        self.m_down3 = self._make_down_stage(128, 256)

        self.m_body = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.m_up3 = self._make_up_stage(256, 128)
        self.m_up2 = self._make_up_stage(128, 64)
        self.m_up1 = self._make_up_stage(64, 32)

        self.m_tail = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _make_down_stage(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create one encoder stage: two residual blocks then downsample."""
        return nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
        )

    @staticmethod
    def _make_up_stage(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create one decoder stage: conv, pixel shuffle, then two residuals."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Sequential(
                nn.PixelShuffle(2), nn.ReLU(inplace=True)
            ),  # pixelshuffl -> espatial upsample by 2, channels reduced by 4 (2^2)
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    @staticmethod
    def _run_down_stage(stage: nn.Sequential, x: Tensor) -> Tensor:
        """Run an encoder stage and return the post-convolution skip tensor."""
        res1, res2, downsample = stage  # same order as defined in _make_down_stage
        return downsample(res2(res1(x)))

    @staticmethod
    def _run_up_stage(stage: nn.Sequential, x: Tensor, skip: Tensor) -> Tensor:
        """Run a decoder stage and add the matching encoder skip tensor."""
        conv, upsample, res1, res2 = (
            stage  # same order as defined in _make_up_stage order
        )
        return res2(res1(upsample(conv(x)))) + skip

    def forward(self, x: Tensor) -> Tensor:
        """Run the denoising network.

        Args:
            x: RGB image batch in ``NCHW`` format.

        Returns:
            Denoised image batch with the same shape as ``x``.
        """
        head = self.m_head(x)  # bchw=b,32,512,512

        down1 = self._run_down_stage(self.m_down1, head)  # bchw=b,64,256,256
        down2 = self._run_down_stage(self.m_down2, down1)  # bchw=b,128,128,128
        down3 = self._run_down_stage(self.m_down3, down2)  # bchw=b,256,64,64

        body = self.m_body(down3)  # bchw=b,256,64,64

        up3 = self._run_up_stage(self.m_up3, body, down2)  # bchw=b,128,128,128
        up2 = self._run_up_stage(self.m_up2, up3, down1)  # bchw=b,64,256,256
        up1 = self._run_up_stage(self.m_up1, up2, head)  # bchw=b,32,512,512

        return (
            x + self.m_tail(up1)
        )  # add the input as a skip connecition, so the model ouputs a denoised image, instead of the noise itself.


class _DenoiserContainer(nn.Module):
    """Container preserving the original checkpoint's module prefix."""

    def __init__(self) -> None:
        """Initialize the wrapped prior network."""
        super().__init__()
        self.prior_net = PriorNet()

    def forward(self, x: Tensor) -> Tensor:
        """Forward to the prior network."""
        return self.prior_net(x)


class DenoiseModel(nn.Module):
    """Top-level denoiser with parameter names matching ``weights.csv``."""

    def __init__(self) -> None:
        """Initialize the denoising model."""
        super().__init__()
        self.model = _DenoiserContainer()

    def forward(self, x: Tensor) -> Tensor:
        """Apply image denoising."""
        return self.model(x)


def _checkpoint_state_dict(payload: Any) -> dict[str, Tensor]:
    """Extract a state dict from either a raw or metadata-wrapped checkpoint."""
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(
        "Checkpoint must be a state_dict or a dict containing 'state_dict'."
    )


def convert_weights_csv_to_checkpoint(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
) -> dict[str, Tensor]:
    """Convert flattened CSV weights into a PyTorch checkpoint.

    The assessment CSV stores convolution weights flattened from
    ``(in_channels, out_channels, kernel_height, kernel_width)`` tensors, while
    PyTorch expects ``(out_channels, in_channels, kernel_height, kernel_width)``.

    Args:
        csv_path: Path to ``weights.csv``.
        checkpoint_path: Destination path for ``checkpoint.pt``.
        model: Model whose ``state_dict`` supplies expected names and shapes.

    Returns:
        Converted PyTorch state dict.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If layer names or element counts do not match the model.
    """
    csv_path = Path(csv_path)
    checkpoint_path = Path(checkpoint_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find weights CSV: {csv_path}")

    model = model or DenoiseModel()
    expected = model.state_dict()

    weights = pd.read_csv(
        csv_path,
        dtype={"layer_name": "category", "index": "int64", "value": "float32"},
    )
    required_columns = {"layer_name", "index", "value"}
    if set(weights.columns) != required_columns:
        raise ValueError(
            f"Expected CSV columns {sorted(required_columns)}, got {list(weights.columns)}"
        )

    csv_layers = set(weights["layer_name"].astype(str).unique())
    expected_layers = set(expected)
    missing = sorted(expected_layers - csv_layers)
    extra = sorted(csv_layers - expected_layers)
    if missing or extra:
        raise ValueError(
            f"Layer mismatch. Missing from CSV: {missing}. Extra in CSV: {extra}."
        )

    converted: dict[str, Tensor] = {}
    for layer_name, group in weights.groupby("layer_name", observed=True):
        name = str(layer_name)
        target_shape = tuple(expected[name].shape)
        if len(target_shape) != 4:
            raise ValueError(
                f"Only 4D convolution weights are supported, got {name}: {target_shape}"
            )

        expected_count = expected[name].numel()
        if len(group) != expected_count:
            raise ValueError(
                f"{name} has {len(group)} values, expected {expected_count}."
            )

        ordered = group.sort_values("index")
        index_values = ordered["index"].to_numpy()
        if not np.array_equal(index_values, np.arange(expected_count)):
            raise ValueError(
                f"{name} indices must be exactly 0..{expected_count - 1} with no gaps or duplicates."
            )

        out_channels, in_channels, kernel_height, kernel_width = target_shape
        csv_shape = (in_channels, out_channels, kernel_height, kernel_width)
        tensor = (
            torch.from_numpy(ordered["value"].to_numpy(copy=True))
            .reshape(csv_shape)
            .permute(1, 0, 2, 3)  # convert to PyTorch's expected layout
            .contiguous()
        )
        converted[name] = tensor

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": converted,
            "metadata": {
                "source": str(csv_path),
                "weight_layout": "csv(in,out,k_h,k_w)->torch(out,in,k_h,k_w)",
            },
        },
        checkpoint_path,
    )
    return converted


def load_model(
    checkpoint_path: str | Path,
    weights_csv_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> DenoiseModel:
    """Load the denoising model from a checkpoint, converting CSV if needed.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint.
        weights_csv_path: Optional CSV path used when the checkpoint is absent.
        device: Target device for the model.

    Returns:
        Loaded model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint_path)
    model = DenoiseModel()

    if not checkpoint_path.exists():
        if weights_csv_path is None:
            raise FileNotFoundError(
                f"Cannot find checkpoint {checkpoint_path}; pass weights_csv_path to create it."
            )
        convert_weights_csv_to_checkpoint(weights_csv_path, checkpoint_path, model)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _checkpoint_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

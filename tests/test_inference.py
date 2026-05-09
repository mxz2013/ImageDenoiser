"""Regression test for denoising inference on the first sample image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evaluation.compute_metrics import compute_pair_metrics, load_rgb_image
from denoiser.model.denoise_model import load_model


def test_inference_improves_metrics_for_first_image() -> None:
    """Check that the denoiser improves PSNR and SSIM on ``00.png``."""
    noisy_path = Path("images/noisy/00.png")
    gt_path = Path("images/gt/00.png")
    checkpoint_path = Path("checkpoint.pt")
    weights_csv_path = Path("weights.csv")

    model = load_model(checkpoint_path, weights_csv_path, device="cpu")
    noisy = load_rgb_image(noisy_path)
    gt = load_rgb_image(gt_path)

    noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).contiguous()
    with torch.no_grad():
        prediction_tensor = model(noisy_tensor).clamp(0.0, 1.0).squeeze(0)

    prediction = prediction_tensor.permute(1, 2, 0).numpy().astype(np.float32)

    assert prediction.shape == gt.shape
    assert prediction.dtype == np.float32
    assert 0.0 <= float(prediction.min()) <= float(prediction.max()) <= 1.0

    noisy_psnr, noisy_ssim = compute_pair_metrics(gt, noisy)
    denoised_psnr, denoised_ssim = compute_pair_metrics(gt, prediction)

    assert denoised_psnr > noisy_psnr
    assert denoised_ssim > noisy_ssim

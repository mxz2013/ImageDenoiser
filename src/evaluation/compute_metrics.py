"""Compute RGB PSNR and SSIM metrics for denoising outputs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as a float array in ``[0, 1]``."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_pair_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM between two RGB images."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")

    psnr = peak_signal_noise_ratio(reference, candidate, data_range=1.0)
    ssim = structural_similarity(reference, candidate, channel_axis=-1, data_range=1.0)
    return float(psnr), float(ssim)


def compute_metrics(
    gt_dir: str | Path = "images/gt",
    noisy_dir: str | Path = "images/noisy",
    denoised_dir: str | Path = "outputs/denoised",
    output_dir: str | Path = "outputs/metrics",
) -> pd.DataFrame:
    """Compute, save, and plot denoising metrics.

    Args:
        gt_dir: Directory containing ground-truth RGB images.
        noisy_dir: Directory containing noisy RGB images.
        denoised_dir: Directory containing predicted denoised RGB images.
        output_dir: Directory for ``metrics.csv`` and ``metrics.png``.

    Returns:
        DataFrame containing per-image rows plus an average row.
    """
    gt_dir = Path(gt_dir)
    noisy_dir = Path(noisy_dir)
    denoised_dir = Path(denoised_dir)
    output_dir = Path(output_dir)

    rows: list[dict[str, float | str]] = []
    for gt_path in sorted(gt_dir.glob("*.png")):
        noisy_path = noisy_dir / gt_path.name
        denoised_path = denoised_dir / gt_path.name
        if not noisy_path.exists():
            raise FileNotFoundError(f"Missing noisy image for {gt_path.name}: {noisy_path}")
        if not denoised_path.exists():
            raise FileNotFoundError(f"Missing denoised image for {gt_path.name}: {denoised_path}")

        gt = load_rgb_image(gt_path)
        noisy = load_rgb_image(noisy_path)
        denoised = load_rgb_image(denoised_path)

        noisy_psnr, noisy_ssim = compute_pair_metrics(gt, noisy)
        denoised_psnr, denoised_ssim = compute_pair_metrics(gt, denoised)
        rows.append(
            {
                "image": gt_path.name,
                "noisy_psnr": noisy_psnr,
                "denoised_psnr": denoised_psnr,
                "psnr_improvement": denoised_psnr - noisy_psnr,
                "noisy_ssim": noisy_ssim,
                "denoised_ssim": denoised_ssim,
                "ssim_improvement": denoised_ssim - noisy_ssim,
            }
        )

    if not rows:
        raise ValueError(f"No ground-truth PNG files found in {gt_dir}")

    dataframe = pd.DataFrame(rows)
    average_row = {"image": "AVERAGE"}
    for column in dataframe.columns:
        if column != "image":
            average_row[column] = float(dataframe[column].mean())
    dataframe = pd.concat([dataframe, pd.DataFrame([average_row])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    plot_path = output_dir / "metrics.png"
    dataframe.to_csv(csv_path, index=False)
    plot_metrics(dataframe[dataframe["image"] != "AVERAGE"], plot_path)

    logger.info(dataframe.tail(1).to_string(index=False))
    logger.info("Saved metrics to %s", csv_path)
    logger.info("Saved plot to %s", plot_path)
    return dataframe


def plot_metrics(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    """Save a PSNR/SSIM comparison plot."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    x = np.arange(len(dataframe))
    width = 0.38
    axes[0].bar(x - width / 2, dataframe["noisy_psnr"], width, label="Noisy")
    axes[0].bar(x + width / 2, dataframe["denoised_psnr"], width, label="Denoised")
    axes[0].set_title("PSNR")
    axes[0].set_xticks(x, dataframe["image"], rotation=45, ha="right")
    axes[0].set_ylabel("dB")
    axes[0].legend()

    axes[1].bar(x - width / 2, dataframe["noisy_ssim"], width, label="Noisy")
    axes[1].bar(x + width / 2, dataframe["denoised_ssim"], width, label="Denoised")
    axes[1].set_title("SSIM")
    axes[1].set_xticks(x, dataframe["image"], rotation=45, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute denoising PSNR and SSIM metrics.")
    parser.add_argument("--gt-dir", default="images/gt", help="Ground-truth image directory.")
    parser.add_argument("--noisy-dir", default="images/noisy", help="Noisy image directory.")
    parser.add_argument("--denoised-dir", default="outputs/denoised", help="Denoised output directory.")
    parser.add_argument("--output-dir", default="outputs/metrics", help="Metrics output directory.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    compute_metrics(args.gt_dir, args.noisy_dir, args.denoised_dir, args.output_dir)


if __name__ == "__main__":
    main()

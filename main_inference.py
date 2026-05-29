"""Run image denoising inference for one image or a directory of images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from denoiser.model.denoise_model import load_model

logger = logging.getLogger(__name__)


class ImageDenoisingDataset(Dataset[tuple[Tensor, str]]):
    """Dataset that reads one RGB image or all PNG/JPEG images in a directory."""

    VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, input_path: str | Path) -> None:
        """Create the image dataset.

        Args:
            input_path: Image file path or directory containing images.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path contains no supported image files.
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        if self.input_path.is_file():
            self.image_paths = [self.input_path]
        else:
            self.image_paths = sorted(
                path
                for path in self.input_path.iterdir()
                if path.is_file() and path.suffix.lower() in self.VALID_SUFFIXES
            )

        if not self.image_paths:
            raise ValueError(f"No supported images found in {self.input_path}")

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        """Load an image as a normalized ``CHW`` float tensor."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(
                f"{image_path.name}: dimensions {w}×{h} must both be divisible by 8."
            )  # 3 downsamplings stages, 512 -> 64
        array = (
            np.asarray(image, dtype=np.float32) / 255.0
        )  # convert uint8 to float32 and norm to [0.0, 1.0]
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        # in general torch model uses troch.float32, but if not we do
        # PIL image / uint8
        # -> NumPy float32 normalized to [0, 1]
        # -> Torch tensor
        # -> move/cast to model device and dtype
        # model_dtype = next(model.parameters()).dtype
        # tensor = tensor.to(device=device, dtype=model_dtype)
        return tensor, image_path.name


def save_image_tensor(tensor: Tensor, output_path: str | Path) -> None:
    """Save a ``CHW`` image tensor as an RGB PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    image.save(output_path)


def run_inference(
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path = "checkpoint.pt",
    weights_csv_path: str | Path | None = "weights.csv",
    batch_size: int = 1,
    n_workers: int = 0,
    device: str | torch.device | None = None,
) -> list[Path]:
    """Denoise images and save outputs.

    Args:
        input_path: Image file path or directory of images.
        output_dir: Directory where denoised PNGs are written.
        checkpoint_path: Path to converted PyTorch checkpoint.
        weights_csv_path: CSV path used to create the checkpoint if missing.
        batch_size: Number of images per inference batch.
        n_workers: Number of DataLoader worker processes (0 = main process only).
        device: Torch device. Defaults to CUDA when available, otherwise CPU.

    Returns:
        Paths to saved denoised images.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(checkpoint_path, weights_csv_path, selected_device)
    dataset = ImageDenoisingDataset(input_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=selected_device.type == "cuda",
    )

    output_dir = Path(output_dir)
    saved_paths: list[Path] = []
    with torch.no_grad():
        for noisy_batch, names in dataloader:
            predictions = model(noisy_batch.to(selected_device)).clamp(
                0.0, 1.0
            )  # the values can go out of range after denoising
            for prediction, name in zip(predictions, names, strict=True):
                output_path = output_dir / Path(name).with_suffix(".png").name
                save_image_tensor(prediction, output_path)
                saved_paths.append(output_path)
                logger.info("  %s → %s", name, output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run denoising inference.")
    parser.add_argument(
        "--input", default="images/noisy", help="Image file or directory to denoise."
    )
    parser.add_argument(
        "--output", default="outputs/denoised", help="Directory for denoised outputs."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoint.pt", help="Path to checkpoint.pt."
    )
    parser.add_argument(
        "--weights-csv", default="weights.csv", help="Path to weights.csv."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Inference batch size."
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, for example 'cpu' or 'cuda'."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    saved_paths = run_inference(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        weights_csv_path=args.weights_csv,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        device=args.device,
    )
    logger.info("Saved %d denoised image(s) to %s", len(saved_paths), Path(args.output))


if __name__ == "__main__":
    main()

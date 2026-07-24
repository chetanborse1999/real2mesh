#!/usr/bin/env python3
"""Track one object through an image sequence and save one mask per image.

With no --box argument, draw a box around the object in the prompt frame and
press Enter or Space. Press C to cancel.

Example:
    python segment_images_sam2.py images masks \
        --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Sequence


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_CHECKPOINT = Path("sam2/checkpoints/sam2.1_hiera_large.pt")
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track one prompted object through an ordered folder of images and "
            "save one binary PNG mask per image."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing frames")
    parser.add_argument("output_folder", type=Path, help="Folder for binary masks")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"SAM2 checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help=(
            "SAM2 Hydra config matching the checkpoint "
            f"(default: {DEFAULT_MODEL_CONFIG})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Object box in the prompt frame; otherwise select it interactively.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Zero-based frame index on which to select the object (default: 0).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in nested input folders.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_folder.is_dir():
        raise ValueError(f"input folder '{args.input_folder}' does not exist")
    if args.input_folder.resolve() == args.output_folder.resolve():
        raise ValueError("input and output folders must be different")
    if not args.checkpoint.is_file():
        raise ValueError(
            f"checkpoint '{args.checkpoint}' does not exist; pass a valid path "
            "with --checkpoint"
        )
    if args.prompt_frame < 0:
        raise ValueError("--prompt-frame must not be negative")
    if args.box is not None:
        x1, y1, x2, y2 = args.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("--box must satisfy X2 > X1 and Y2 > Y1")


def natural_key(path: Path) -> list[int | str]:
    """Sort names naturally so frame_2 precedes frame_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def find_images(folder: Path, recursive: bool) -> list[Path]:
    candidates: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_box_interactively(image_path: Path) -> tuple[float, float, float, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for interactive selection; install "
            "opencv-python or pass --box X1 Y1 X2 Y2"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not open prompt image '{image_path}'")

    window_name = "Select object, then press Enter/Space (C cancels)"
    try:
        x, y, width, height = cv2.selectROI(
            window_name,
            image,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise RuntimeError("object selection was cancelled or empty")
    return float(x), float(y), float(x + width), float(y + height)


def validate_box_in_image(
    box: Sequence[float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"box {tuple(box)} is outside the prompt image bounds "
            f"(width={width}, height={height})"
        )
    return float(x1), float(y1), float(x2), float(y2)


def stage_video_frames(
    image_paths: Sequence[Path],
    staging_folder: Path,
    image_module: Any,
    image_ops: Any,
) -> tuple[int, int]:
    """Create the numerically named JPEG sequence required by SAM2."""
    expected_size: tuple[int, int] | None = None

    for index, source in enumerate(image_paths):
        with image_module.open(source) as opened:
            frame = image_ops.exif_transpose(opened).convert("RGB")
            if expected_size is None:
                expected_size = frame.size
            elif frame.size != expected_size:
                raise ValueError(
                    f"frame '{source}' has size {frame.size}, expected {expected_size}"
                )
            frame.save(staging_folder / f"{index:08d}.jpg", quality=95)

    if expected_size is None:
        raise ValueError("no frames were staged")
    return expected_size


def collect_predictions(
    predictor: Any,
    inference_state: dict[str, Any],
    prompt_frame: int,
) -> dict[int, Any]:
    predictions: dict[int, Any] = {}

    for frame_idx, _, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=prompt_frame,
    ):
        predictions[frame_idx] = (mask_logits[0] > 0.0).to(device="cpu")

    if prompt_frame > 0:
        for frame_idx, _, mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_frame,
            reverse=True,
        ):
            predictions[frame_idx] = (mask_logits[0] > 0.0).to(device="cpu")

    return predictions


def save_one_mask_per_frame(
    predictions: dict[int, Any],
    image_paths: Sequence[Path],
    input_folder: Path,
    output_folder: Path,
    image_module: Any,
    numpy: Any,
) -> None:
    for frame_idx, source in enumerate(image_paths):
        if frame_idx not in predictions:
            raise RuntimeError(f"SAM2 produced no mask for frame {frame_idx}")

        mask = (
            predictions[frame_idx]
            .squeeze()
            .to(device="cpu")
            .numpy()
            .astype(numpy.uint8)
            * 255
        )
        # COLMAP 3.8 expects the complete image filename plus ".png", e.g.
        # "frame_000001.jpg.png" for "frame_000001.jpg".
        relative = source.relative_to(input_folder)
        destination = output_folder / Path(f"{relative}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image_module.fromarray(mask).save(destination)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        image_paths = find_images(args.input_folder, args.recursive)
        if not image_paths:
            raise ValueError(f"no supported images found in '{args.input_folder}'")
        if args.prompt_frame >= len(image_paths):
            raise ValueError(
                f"--prompt-frame {args.prompt_frame} is outside the sequence of "
                f"{len(image_paths)} images"
            )

        try:
            import numpy as np
            from PIL import Image, ImageOps
            import torch

            # Prefer an adjacent cloned SAM2 repository over the namespace
            # package created by running beside a folder also named `sam2`.
            local_sam2_repository = Path(__file__).resolve().parent / "sam2"
            if (local_sam2_repository / "sam2" / "__init__.py").is_file():
                sys.path.insert(0, str(local_sam2_repository))

            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                f"missing dependency '{exc.name}'; install PyTorch, Pillow, and "
                "the official SAM2 package"
            ) from exc

        prompt_path = image_paths[args.prompt_frame]
        with Image.open(prompt_path) as prompt_image:
            prompt_size = ImageOps.exif_transpose(prompt_image).size

        selected_box = (
            tuple(args.box)
            if args.box is not None
            else select_box_interactively(prompt_path)
        )
        box = validate_box_in_image(selected_box, prompt_size)
        print(f"Tracking box {box} from frame {args.prompt_frame}: {prompt_path}")

        device = choose_device(torch, args.device)
        print(f"Loading SAM2 video predictor on {device}...")
        predictor = build_sam2_video_predictor(
            args.model_config,
            str(args.checkpoint),
            device=device,
            apply_postprocessing=False,
        )

        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as temporary:
            staging_folder = Path(temporary)
            print(f"Preparing {len(image_paths)} ordered frames...")
            stage_video_frames(image_paths, staging_folder, Image, ImageOps)

            precision_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), precision_context:
                state = predictor.init_state(
                    video_path=str(staging_folder),
                    offload_video_to_cpu=True,
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=args.prompt_frame,
                    obj_id=1,
                    box=np.asarray(box, dtype=np.float32),
                )
                predictions = collect_predictions(
                    predictor,
                    state,
                    args.prompt_frame,
                )

            save_one_mask_per_frame(
                predictions,
                image_paths,
                args.input_folder,
                args.output_folder,
                Image,
                np,
            )

        print(
            f"Saved {len(image_paths)} masks to '{args.output_folder}' "
            "(one mask per image)."
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Extract frames from a video file at a specified frame rate using OpenCV.

Usage:
    python extract_frames.py --video_file <video_file> --frame_rate <frame_rate> [--output_dir <output_dir>]

Arguments:
    --video_file: Path to the input video file (required)
    --frame_rate: Frame rate (frames per second) to extract (required)
    --output_dir: Output directory to save frames (default: ./frames)
"""

import cv2
import os
from pathlib import Path
import shutil
import sys
import argparse


def prepare_output_directory(output_dir):
    """Create the output directory, removing contents from any previous run."""
    output_path = Path(output_dir).resolve()
    current_path = Path.cwd().resolve()

    if output_path == Path(output_path.anchor) or output_path == current_path:
        raise ValueError(
            f"Refusing to empty unsafe output directory '{output_path}'"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    for child in output_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    print(f"Prepared empty output directory: {output_path}")
    return str(output_path)


def extract_frames(video_path, frame_rate, output_dir="frames"):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the input video file
        frame_rate: Frame rate (frames per second) to extract
        output_dir: Directory to save extracted frames
    
    Returns:
        True if successful, False otherwise
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return False

    # Start each valid extraction run with an empty output directory.
    try:
        output_dir = prepare_output_directory(output_dir)
    except (OSError, ValueError) as exc:
        cap.release()
        print(f"Error: Cannot prepare output directory: {exc}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Original FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Target extraction rate: {frame_rate} fps")
    
    # Calculate frame interval
    frame_interval = int(fps / frame_rate) if frame_rate > 0 else 1
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"  Extracting every {frame_interval} frame(s)")
    
    frame_count = 0
    extracted_count = 0
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                print(f"Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Total frames extracted: {extracted_count}")
    print(f"Frames saved to: {output_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file using OpenCV"
    )
    parser.add_argument(
        "--video_file",
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        required=True,
        help="Frame rate (frames per second) to extract"
    )
    parser.add_argument(
        "--output_dir",
        default="images",
        help="Output directory for frames (default: frames)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_file):
        print(f"Error: Video file '{args.video_file}' not found")
        sys.exit(1)
    
    if args.frame_rate <= 0:
        print("Error: Frame rate must be greater than 0")
        sys.exit(1)
    
    # Extract frames
    success = extract_frames(args.video_file, args.frame_rate, args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

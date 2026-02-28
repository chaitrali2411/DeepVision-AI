"""Extract frames from video for dataset (injection / iv_change / no_event). No raw video stored."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 30,
    max_frames: int | None = 1000,
) -> int:
    """Extract every Nth frame; save under output_dir. Returns count saved."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = 0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                out_path = output_dir / f"frame_{count:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from video into a class folder")
    p.add_argument("video", help="Path to video file")
    p.add_argument("output_dir", help="Output directory (e.g. dataset/injection)")
    p.add_argument("--every-n", type=int, default=30, help="Save every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames to save")
    args = p.parse_args()
    n = extract_frames(args.video, args.output_dir, every_n=args.every_n, max_frames=args.max_frames)
    print(f"Saved {n} frames to {args.output_dir}")


if __name__ == "__main__":
    main()

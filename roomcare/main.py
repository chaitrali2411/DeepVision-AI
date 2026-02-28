"""RoomCare entry point: real-time webcam or video inference. Observe → Interpret → Update Memory → Evaluate → Act."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as python main.py from inside roomcare/
_roomcare_dir = Path(__file__).resolve().parent
_parent = _roomcare_dir.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from roomcare.agent import RoomCareAgent
from roomcare.config import DATA_DIR, FRAME_SKIP, PROJECT_ROOT, WEBCAM_ID


def run_webcam(agent: RoomCareAgent, frame_skip: int = FRAME_SKIP, camera_id: int = WEBCAM_ID) -> None:
    """Run agent on webcam; no raw video stored."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                event_type = agent.step_frame(pil)
                cv2.putText(
                    frame,
                    event_type,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            frame_idx += 1
            cv2.imshow("RoomCare", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_video(
    agent: RoomCareAgent,
    video_path: str | Path,
    frame_skip: int = FRAME_SKIP,
    output_path: str | Path | None = None,
) -> dict:
    """Run agent on video file; no raw video stored. Returns events + alerts for optional output file."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    start_ts = time.time()
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                agent.step_frame(pil)
            frame_idx += 1
    finally:
        cap.release()

    events = agent.memory.get_events_since(start_ts)
    alerts = agent.memory.get_alerts_since(start_ts)
    result = {"video": str(video_path), "frames_processed": frame_idx, "events": events, "alerts": alerts}
    print(f"Processed {frame_idx} frames (every {frame_skip}th classified). Events: {len(events)}, Alerts: {len(alerts)}.")

    if output_path:
        import json
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {output_path}")

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="RoomCare: video in → results out (camera only if you pass --webcam)")
    p.add_argument("--video", type=str, default=None, help="Input video path (default: use this for inference)")
    p.add_argument("--output", type=str, default=None, help="Write results (events + alerts) to this JSON file")
    p.add_argument("--webcam", action="store_true", help="Use live camera instead of video file (camera will open only if set)")
    p.add_argument("--frame_skip", type=int, default=FRAME_SKIP, help="Run vision every N frames")
    p.add_argument("--camera_id", type=int, default=WEBCAM_ID, help="Webcam device id (only if --webcam)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA adapter (optional)")
    args = p.parse_args()

    from roomcare.perception import VisionClassifier
    vision = VisionClassifier(lora_adapter=args.checkpoint) if args.checkpoint else None
    agent = RoomCareAgent(vision=vision)

    if args.webcam:
        run_webcam(agent, frame_skip=args.frame_skip, camera_id=args.camera_id)
    elif args.video:
        run_video(agent, args.video, frame_skip=args.frame_skip, output_path=args.output)
    else:
        p.print_help()
        print("\nUsage: --video path/to/input.mp4 [--output path/to/results.json]")
        print("       (Camera is not used unless you pass --webcam)")


if __name__ == "__main__":
    main()

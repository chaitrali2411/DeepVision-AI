"""RoomCare configuration. No raw video/audio paths; only structured data."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"       # Training only (injection, iv_change, no_event)
INPUT_DIR = PROJECT_ROOT / "input"        # User input videos for inference
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DB_PATH = PROJECT_ROOT / "roomcare.db"

# Event labels (vision classification)
EVENT_LABELS = ("injection_event", "iv_change_event", "no_event")
LABEL_TO_ID = {l: i for i, l in enumerate(EVENT_LABELS)}
ID_TO_LABEL = {i: l for i, l in enumerate(EVENT_LABELS)}

# Dataset folder names (must match your dirs: injection/, iv_change/, no_event/)
DATASET_FOLDERS = ("injection", "iv_change", "no_event")
FOLDER_TO_LABEL = {"injection": "injection_event", "iv_change": "iv_change_event", "no_event": "no_event"}

# Vision model: Gemma 3n (multimodal). Requires Hugging Face login + accept model terms.
GEMMA_MODEL = "google/gemma-3n-e2b-it"
GEMMA_VISION_PROMPT = (
    "Classify this hospital room image. Reply with exactly one of: injection_event, iv_change_event, no_event. "
    "Answer only the label, nothing else."
)

# Inference
FRAME_SKIP = 15  # run vision every N frames
WEBCAM_ID = 0
MAX_NEW_TOKENS = 32
DEVICE = "cuda"  # set to "cpu" if no GPU

# Memory / reasoning
ROLLING_WINDOW_HOURS = 24
MIN_SAMPLES_FOR_STATS = 5
ANOMALY_Z_SCORE_CAP = 4.0  # cap z for robustness

# Audio (on-device Whisper)
WHISPER_MODEL = "tiny"  # tiny/base/small for on-device
SAMPLE_RATE = 16000

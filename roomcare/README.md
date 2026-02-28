# RoomCare — On-Device Multimodal AI Agent

Hospital room care event observer: perception (vision + audio), memory (SQLite), reasoning (rolling stats, adaptive anomaly detection), and actions — **fully on-device**. No raw video/audio is stored; only structured metadata persists.

## Purpose

- Observe bedside care events: **injection**, **IV replacement**
- Maintain a structured, time-aware ledger
- Autonomously detect unusual patterns via statistical deviation (no fixed thresholds)

## Architecture

```
Observe → Interpret → Update Memory → Evaluate Pattern → Act
```

- **Perception**: Vision classification (Gemma 3n), on-device audio transcription, intent extraction
- **Fusion**: Align voice + vision events, assign verification level
- **Memory**: SQLite (`care_events`, `alerts`)
- **Reasoning**: Rolling statistics (frequency, intervals), adaptive anomaly detection

## Setup

**Gemma 3n is gated.** Before running:
1. Accept the model terms: open https://huggingface.co/google/gemma-3n-e2b-it → sign in → click **Agree and access repository**.
2. Create a token: https://huggingface.co/settings/tokens → **Create new token** (Role: Read) → copy the token.
3. Log in in the terminal: `pip install -U huggingface_hub && huggingface-cli login` → paste your token when prompted.

```bash
cd roomcare
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Two-phase workflow

### Phase 1: Training (use only dataset — do not use input videos here)

The model is trained on **all three** event types from `dataset/`:

| Folder              | Label            | Purpose                          |
|---------------------|------------------|----------------------------------|
| `dataset/injection/`  | injection_event  | Videos of injection care         |
| `dataset/iv_change/`  | iv_change_event | Videos of IV bag/line change      |
| `dataset/no_event/`   | no_event        | Videos with no injection/IV change |

- Training uses **whatever videos you put in these three folders**. The model learns to predict one of: `injection_event`, `iv_change_event`, `no_event`.
- If a folder is empty (e.g. `no_event/`), that class has no training data and the model will be weaker on it. **For best results, add videos to all three folders**, then run training.
- Do **not** use your “input” or test videos inside `dataset/` for training — keep `dataset/` for **training-only** videos.

### Phase 2: Inference (user input → result)

- Put **any** video you want to analyze in `roomcare/input/` (or pass any path).
- Run the agent on that video; results go to `--output` (e.g. `results.json`).
- Training videos (e.g. `injection1.mp4` in `dataset/injection/`) should **not** be the only thing you run inference on — use **new** videos in `input/` (or elsewhere) to get real-world results.

## Folder layout

```
roomcare/
  dataset/          ← Training only (injection, iv_change, no_event videos)
    injection/
    iv_change/
    no_event/
  input/            ← User input videos for inference (any video to classify)
  checkpoints/      ← Saved model after training
```

## Usage

Run from the **parent** of `roomcare` (e.g. `DeepVision_AI`):

**1. Train** (on dataset only; include all three classes for best results):
  ```bash
  python -m roomcare.train --data_dir roomcare/dataset --output_dir roomcare/checkpoints/roomcare_lora
  ```

**2. Run inference** on a user video (e.g. from `input/` or any path):
  ```bash
  python -m roomcare --video roomcare/input/my_video.mp4 --output results.json --checkpoint roomcare/checkpoints/roomcare_lora
  ```
  Omit `--checkpoint` to use the base model only.

**3. Webcam** (optional):
  ```bash
  python -m roomcare --webcam
  ```

**4. Extract frames** (to build dataset from raw videos):
  ```bash
  python roomcare/tools/extract_frames.py path/to/video.mp4 roomcare/dataset/injection --every-n 30
  ```

## Constraints

- Raw video/audio are **not** stored
- Only structured metadata (events, alerts, stats) is persisted
- Fully on-device; GPU used when available

"""LoRA fine-tuning for RoomCare vision classifier (Gemma 3n). Dataset: videos or images in dataset/injection, iv_change, no_event."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

_roomcare_dir = Path(__file__).resolve().parent
_parent = _roomcare_dir.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from roomcare.config import (
    DATA_DIR,
    EVENT_LABELS,
    GEMMA_MODEL,
    LABEL_TO_ID,
    PROJECT_ROOT,
    DATASET_FOLDERS,
    FOLDER_TO_LABEL,
)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class RoomCareVideoDataset(Dataset):
    """Dataset from videos in dataset/injection, iv_change, no_event. Extracts frames on-the-fly (no photos needed)."""

    def __init__(self, data_dir: Path, every_n_frames: int = 30, max_frames_per_video: int = 200):
        self.data_dir = Path(data_dir)
        self.every_n_frames = every_n_frames
        self.max_frames_per_video = max_frames_per_video
        self.samples = []  # list of (video_path, frame_index, label_id)
        for folder_name in DATASET_FOLDERS:
            label = FOLDER_TO_LABEL[folder_name]
            folder = self.data_dir / folder_name
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    continue
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total <= 0:
                    continue
                indices = list(range(0, total, every_n_frames))[:max_frames_per_video]
                for idx in indices:
                    self.samples.append((str(path), idx, LABEL_TO_ID[label]))
        self._cap = None
        self._current_path = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        video_path, frame_idx, label = self.samples[i]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = __import__("numpy").zeros((224, 224, 3), dtype="uint8")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return {"image": img, "label": label, "path": video_path}


class RoomCareDataset(Dataset):
    """Image dataset from dataset/injection, iv_change, no_event (optional)."""

    def __init__(self, data_dir: Path, processor=None, transform=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.transform = transform
        self.samples = []
        for folder_name in DATASET_FOLDERS:
            label = FOLDER_TO_LABEL[folder_name]
            folder = self.data_dir / folder_name
            if not folder.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for path in folder.glob(ext):
                    self.samples.append((str(path), LABEL_TO_ID[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label, "path": path}


def _collate_fn(batch):
    """Keep PIL images as list; default_collate cannot stack PIL Images."""
    return {
        "image": [b["image"] for b in batch],
        "label": [b["label"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def _count_videos(data_dir: Path) -> int:
    n = 0
    for folder_name in DATASET_FOLDERS:
        folder = data_dir / folder_name
        if folder.is_dir():
            for p in folder.iterdir():
                if p.suffix.lower() in VIDEO_EXTENSIONS:
                    n += 1
    return n


def train_lora(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    model_name: str = GEMMA_MODEL,
    epochs: int = 3,
    batch_size: int = 1,
    lr: float = 2e-5,
    every_n_frames: int = 30,
    max_frames_per_video: int = 200,
) -> None:
    """LoRA fine-tune Gemma 3n for 3-class classification (image + prompt -> label)."""
    data_dir = Path(data_dir or DATA_DIR)
    output_dir = Path(output_dir or PROJECT_ROOT / "checkpoints" / "roomcare_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",  # vision tower (TimmWrapper) does not support sdpa
    )
    if device == "cpu":
        model = model.to("cpu")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if _count_videos(data_dir) > 0:
        dataset = RoomCareVideoDataset(
            data_dir, every_n_frames=every_n_frames, max_frames_per_video=max_frames_per_video
        )
        print(f"Training from videos: {len(dataset)} frames from {data_dir}")
    else:
        dataset = RoomCareDataset(data_dir, processor=processor)
        print(f"Training from images: {len(dataset)} images from {data_dir}")
    if len(dataset) == 0:
        print("No data found. Put videos (or images) in dataset/injection/, dataset/iv_change/, dataset/no_event/")
        return

    prompt = "Classify this hospital room image. Reply with exactly one of: injection_event, iv_change_event, no_event. Answer only the label."
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate_fn
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"]
            labels = batch["label"]
            if not isinstance(images, (list, tuple)):
                images = [images]
            images = [img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in images]
            messages_list = []
            for img, label in zip(images, labels):
                label_text = EVENT_LABELS[label]
                messages_list.append([
                    {"role": "system", "content": [{"type": "text", "text": "You are a hospital room image classifier."}]},
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": label_text}]},
                ])
            inputs = processor.apply_chat_template(
                messages_list,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels_in = inputs.get("labels")
            if labels_in is None:
                labels_in = inputs["input_ids"].clone()
                pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
                labels_in[labels_in == pad_id] = -100
            outputs = model(**inputs, labels=labels_in)
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="Dataset root (default: config DATA_DIR)")
    p.add_argument("--output_dir", default=None, help="Checkpoint output dir")
    p.add_argument("--model", default=GEMMA_MODEL, help="Base model (Gemma 3n)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1, help="Keep 1 for chat template with images")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--every_n_frames", type=int, default=30, help="For video dataset: sample every N frames")
    p.add_argument("--max_frames_per_video", type=int, default=200, help="Max frames per video")
    args = p.parse_args()
    train_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        every_n_frames=args.every_n_frames,
        max_frames_per_video=args.max_frames_per_video,
    )


if __name__ == "__main__":
    main()

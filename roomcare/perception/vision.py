"""Vision classification via Gemma 3n: injection_event, iv_change_event, no_event."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from roomcare.config import DEVICE, EVENT_LABELS, GEMMA_MODEL, GEMMA_VISION_PROMPT

MAX_NEW_TOKENS = 32


class VisionClassifier:
    """On-device frame classification using Gemma 3n (vision). Uses chat template so image tokens match features."""

    def __init__(
        self,
        model_name: str = GEMMA_MODEL,
        device: str | None = None,
        prompt: str = GEMMA_VISION_PROMPT,
        lora_adapter: str | Path | None = None,
    ):
        self.model_name = model_name
        self.device = device or DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.prompt = prompt
        self.lora_adapter = Path(lora_adapter) if lora_adapter else None
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        self._model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager",  # TimmWrapperModel (vision tower) does not support sdpa yet
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        if self.lora_adapter and self.lora_adapter.exists():
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, str(self.lora_adapter))
            except Exception:
                pass

    def classify(self, image: Image.Image | str) -> str:
        """Return one of: injection_event, iv_change_event, no_event."""
        self._ensure_loaded()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Gemma 3n processor expects text to contain image_token so it expands to 256 image tokens (matches features)
        image_token = getattr(self._processor, "image_token", "<image>")
        text_with_placeholder = f"{image_token} {self.prompt}"
        inputs = self._processor(
            images=[image],
            text=[text_with_placeholder],
            return_tensors="pt",
        )
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        text = self._processor.decode(out[0], skip_special_tokens=True)
        return self._parse_label(text)

    def _parse_label(self, text: str) -> str:
        text = text.strip().lower()
        for label in EVENT_LABELS:
            if label in text or label.replace("_", " ") in text:
                return label
        if "injection" in text:
            return "injection_event"
        if "iv_change" in text or "iv change" in text:
            return "iv_change_event"
        return "no_event"

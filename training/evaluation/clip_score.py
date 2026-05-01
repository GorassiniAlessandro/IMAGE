from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class ClipScoreResult:
    checkpoint_name: str
    image_path: str
    prompt: str
    score: float


class ClipScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score_pair(self, image_path: str | Path, prompt: str) -> float:
        with Image.open(image_path) as image:
            inputs = self.processor(text=[prompt], images=image.convert("RGB"), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embeds * text_embeds).sum(dim=-1).item())

    @torch.no_grad()
    def score_batch(self, image_paths: Iterable[str | Path], prompts: Iterable[str]) -> list[float]:
        scores: list[float] = []
        for image_path, prompt in zip(image_paths, prompts):
            scores.append(self.score_pair(image_path, prompt))
        return scores


def write_clip_results_json(results: Iterable[ClipScoreResult], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_clip_results_csv(results: Iterable[ClipScoreResult], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint_name", "image_path", "prompt", "score"])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path

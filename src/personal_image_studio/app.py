from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import psutil
import random
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
  from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional dependency fallback
  Image = None
  ImageDraw = None

try:
  import torch
except Exception:  # pragma: no cover - optional dependency fallback
  torch = None

try:
  from diffusers import StableDiffusionPipeline
except Exception:  # pragma: no cover - optional dependency fallback
  StableDiffusionPipeline = None

try:
  from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency fallback
  GoogleTranslator = None


ProviderName = Literal["mock", "fooocus", "local"]

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "studio_config.json"
TRAINING_RAW_DIR = ROOT_DIR / "training" / "dataset" / "raw"
PERSONAL_LORA_DEFAULT_PATH = ROOT_DIR / "training" / "output" / "personal-lora" / "pytorch_lora_weights.safetensors"

DEFAULT_PROFILES: dict[str, dict[str, object]] = {
  "default": {
    "name": "default",
    "display_name": "Default",
    "style": "Default",
    "aspect_ratio": "1024x1024",
    "count": 1,
    "notes": "Profilo generale per test rapidi e immagini pulite.",
  },
  "cinematic": {
    "name": "cinematic",
    "display_name": "Cinematic",
    "style": "Cinematic",
    "aspect_ratio": "1344x768",
    "count": 1,
    "notes": "Perfetto per concept, poster e scene narrative.",
  },
  "portrait": {
    "name": "portrait",
    "display_name": "Portrait",
    "style": "Photographic",
    "aspect_ratio": "896x1152",
    "count": 1,
    "notes": "Buono per volti, avatar e ritratti verticali.",
  },
  "batch": {
    "name": "batch",
    "display_name": "Batch",
    "style": "Concept Art",
    "aspect_ratio": "1024x1024",
    "count": 4,
    "notes": "Crea più varianti con un solo invio.",
  },
}

STYLE_PRESETS: dict[str, dict[str, str]] = {
  "Default": {"prefix": "", "suffix": "", "negative": ""},
  "Photographic": {
    "prefix": "professional photo, realistic lighting, highly detailed",
    "suffix": "85mm lens, natural skin texture",
    "negative": "cartoon, cgi, low detail",
  },
  "Cinematic": {
    "prefix": "cinematic composition, dramatic lighting, film still",
    "suffix": "anamorphic look, color grading, volumetric light",
    "negative": "flat lighting, low contrast",
  },
  "Anime": {
    "prefix": "anime style, clean lineart, vibrant colors",
    "suffix": "studio quality illustration",
    "negative": "photorealistic, blurry lineart",
  },
  "Manga": {
    "prefix": "manga style, black and white, dynamic framing",
    "suffix": "inked details, screen tones",
    "negative": "color, photorealistic",
  },
  "Concept Art": {
    "prefix": "concept art, production design, ideation",
    "suffix": "high detail environment, mood exploration",
    "negative": "unfinished sketch, noisy",
  },
  "Fantasy Art": {
    "prefix": "epic fantasy art, magical atmosphere",
    "suffix": "ornate details, grand composition",
    "negative": "modern mundane setting",
  },
  "Sci-Fi": {
    "prefix": "science fiction aesthetic, futuristic design",
    "suffix": "advanced materials, cinematic scale",
    "negative": "medieval style",
  },
  "Cyberpunk": {
    "prefix": "cyberpunk neon city, rain reflections, moody",
    "suffix": "high contrast, chrome surfaces",
    "negative": "pastel minimalism",
  },
  "Pixel Art": {
    "prefix": "pixel art, 16-bit style",
    "suffix": "clean pixel grid, retro game palette",
    "negative": "smooth shading, photoreal",
  },
  "Watercolor": {
    "prefix": "watercolor painting",
    "suffix": "soft pigment bleed, textured paper",
    "negative": "hard edges, digital noise",
  },
  "Oil Painting": {
    "prefix": "oil painting style",
    "suffix": "visible brush strokes, rich texture",
    "negative": "flat digital",
  },
  "3D Render": {
    "prefix": "high quality 3d render",
    "suffix": "global illumination, physically based materials",
    "negative": "hand-drawn look",
  },
  "Low Poly": {
    "prefix": "low poly 3d style",
    "suffix": "geometric shapes, flat shading",
    "negative": "high poly photoreal",
  },
  "Isometric": {
    "prefix": "isometric illustration",
    "suffix": "clean geometry, balanced composition",
    "negative": "perspective distortion",
  },
  "Architectural": {
    "prefix": "architectural visualization",
    "suffix": "precise structure, materials realism",
    "negative": "deformed geometry",
  },
  "Product Shot": {
    "prefix": "studio product photography",
    "suffix": "clean background, softbox lighting",
    "negative": "cluttered background",
  },
  "Portrait Studio": {
    "prefix": "portrait studio photo",
    "suffix": "sharp eyes, natural skin tones",
    "negative": "deformed face, extra limbs",
  },
  "Noir": {
    "prefix": "film noir aesthetic",
    "suffix": "moody shadows, monochrome palette",
    "negative": "saturated cheerful colors",
  },
  "Minimal": {
    "prefix": "minimalist design",
    "suffix": "simple shapes, negative space",
    "negative": "visual clutter",
  },
}

AVAILABLE_STYLES: list[str] = list(STYLE_PRESETS.keys())

ITALIAN_HINT_WORDS = {
  "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "a", "da", "in", "con", "su",
  "per", "tra", "fra", "e", "o", "ma", "non", "che", "come", "ritratto", "stile", "dettagli",
  "luce", "notte", "giorno", "futuristico", "cartone", "anime", "sfondo", "qualita",
}


def _looks_like_italian(text: str) -> bool:
  lowered = text.lower()
  if any(char in lowered for char in "àèéìòù"):
    return True
  tokens = re.findall(r"[a-zA-Z]+", lowered)
  if not tokens:
    return False
  matches = sum(1 for token in tokens if token in ITALIAN_HINT_WORDS)
  return matches >= 2


def _translate_to_english(text: str) -> str:
  source = text.strip()
  if not source:
    return text
  if GoogleTranslator is None:
    return text
  if source.isascii() and not _looks_like_italian(source):
    return text
  try:
    translated = GoogleTranslator(source="auto", target="en").translate(source)
    return translated if translated else text
  except Exception:
    # Never block generation if translation fails.
    return text


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    profile: str = Field(default="default", max_length=80)
    style: str = Field(default="Default", max_length=120)
    aspect_ratio: str = Field(default="1024x1024", max_length=20)
    count: int = Field(default=1, ge=1, le=4)
    seed: int | None = Field(default=None)
    creative_note: str = Field(default="", max_length=2000)
    use_personal_model: bool = Field(default=True)


class ImageItem(BaseModel):
    title: str
    image_data_uri: str
    prompt: str
    notes: str


class GenerateResponse(BaseModel):
    provider: ProviderName
    profile: str
    items: list[ImageItem]


class ProfileDefinition(BaseModel):
    name: str
    display_name: str
    style: str
    aspect_ratio: str
    count: int = Field(ge=1, le=4)
    notes: str = ""


class StudioConfig(BaseModel):
    provider: ProviderName = "mock"
    fooocus_endpoint: str = "http://127.0.0.1:7865"
    local_model_id: str = "runwayml/stable-diffusion-v1-5"
    local_device: str = "auto"
    local_num_inference_steps: int = Field(default=25, ge=1, le=100)
    local_guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    local_use_personal_lora: bool = True
    local_personal_lora_path: str = str(PERSONAL_LORA_DEFAULT_PATH)
    local_personal_lora_scale: float = Field(default=1.0, ge=0.1, le=2.0)
    profiles: dict[str, ProfileDefinition] = Field(default_factory=dict)


class ConfigUpdate(BaseModel):
    provider: ProviderName | None = None
    fooocus_endpoint: str | None = None
    local_model_id: str | None = None
    local_device: str | None = None
    local_num_inference_steps: int | None = Field(default=None, ge=1, le=100)
    local_guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    local_use_personal_lora: bool | None = None
    local_personal_lora_path: str | None = None
    local_personal_lora_scale: float | None = Field(default=None, ge=0.1, le=2.0)
    profiles: dict[str, ProfileDefinition] | None = None


@dataclass(slots=True)
class GenerationContext:
    prompt: str
    negative_prompt: str
    creative_note: str
    profile: str
    style: str
    aspect_ratio: str
    seed: int | None
    count: int
    use_personal_model: bool


class ImageProvider:
    provider_name: ProviderName = "mock"

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        raise NotImplementedError


class MockImageProvider(ImageProvider):
    provider_name: ProviderName = "mock"

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        items: list[ImageItem] = []
        for index in range(context.count):
            title = f"Render {index + 1}"
            accent = self._accent_color(context.prompt, index)
            if Image is not None and ImageDraw is not None:
                image_data_uri = self._build_png_data_uri(
                    title=title,
                    prompt=context.prompt,
                    style=context.style,
                    aspect_ratio=context.aspect_ratio,
                    accent=accent,
                    index=index,
                )
                notes = _build_preview_notes(context) + " | Render mode: local-png"
            else:
                svg = self._build_svg(
                    title=title,
                    prompt=context.prompt,
                    style=context.style,
                    aspect_ratio=context.aspect_ratio,
                    accent=accent,
                )
                encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
                image_data_uri = f"data:image/svg+xml;base64,{encoded}"
                notes = _build_preview_notes(context) + " | Render mode: svg-fallback"
            items.append(
                ImageItem(
                    title=title,
                    image_data_uri=image_data_uri,
                    prompt=context.prompt,
                    notes=notes,
                )
            )
        return items

    @staticmethod
    def _build_png_data_uri(title: str, prompt: str, style: str, aspect_ratio: str, accent: str, index: int) -> str:
        seed = MockImageProvider._seed_from_prompt(prompt, style, aspect_ratio, index)
        rng = random.Random(seed)
        width, height = MockImageProvider._parse_aspect_ratio(aspect_ratio)
        width = max(512, min(width, 1536))
        height = max(512, min(height, 1536))

        image = Image.new("RGB", (width, height), "#080b16")
        draw = ImageDraw.Draw(image, "RGBA")

        accent_rgb = MockImageProvider._hex_to_rgb(accent)

        for band in range(9):
            alpha = 30 + band * 8
            offset = int((band / 9) * height)
            color = (accent_rgb[0], accent_rgb[1], accent_rgb[2], alpha)
            draw.rectangle([(0, offset), (width, offset + height // 6)], fill=color)

        for _ in range(26):
            x0 = rng.randint(-width // 4, width)
            y0 = rng.randint(-height // 4, height)
            radius = rng.randint(height // 12, height // 3)
            tone = (
                min(255, accent_rgb[0] + rng.randint(-35, 35)),
                min(255, accent_rgb[1] + rng.randint(-35, 35)),
                min(255, accent_rgb[2] + rng.randint(-35, 35)),
                rng.randint(35, 95),
            )
            draw.ellipse([(x0, y0), (x0 + radius, y0 + radius)], fill=tone)

        frame_color = (255, 255, 255, 40)
        draw.rounded_rectangle(
            [(24, 24), (width - 24, height - 24)],
            radius=24,
            outline=frame_color,
            width=2,
        )

        draw.rounded_rectangle(
            [(40, height - 130), (width - 40, height - 40)],
            radius=18,
            fill=(8, 12, 24, 170),
            outline=(255, 255, 255, 50),
            width=1,
        )

        footer = f"{title} | {style} | {aspect_ratio}"
        prompt_short = (prompt[:140] + "...") if len(prompt) > 140 else prompt
        draw.text((56, height - 112), footer, fill=(236, 242, 255, 230))
        draw.text((56, height - 82), prompt_short, fill=(188, 197, 215, 230))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _seed_from_prompt(prompt: str, style: str, aspect_ratio: str, index: int) -> int:
        source = f"{prompt}|{style}|{aspect_ratio}|{index}"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
        try:
            width_str, height_str = aspect_ratio.lower().split("x", maxsplit=1)
            return int(width_str), int(height_str)
        except Exception:
            return 1024, 1024

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        if len(color) != 6:
            return 34, 197, 94
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    @staticmethod
    def _accent_color(prompt: str, index: int) -> str:
        value = sum(ord(char) for char in prompt) + index * 7919
        palette = ["#f97316", "#22c55e", "#06b6d4", "#e879f9", "#f43f5e", "#facc15"]
        return palette[value % len(palette)]

    @staticmethod
    def _build_svg(title: str, prompt: str, style: str, aspect_ratio: str, accent: str) -> str:
        safe_prompt = html.escape(prompt)
        safe_style = html.escape(style)
        safe_ratio = html.escape(aspect_ratio)
        return f"""<svg xmlns='http://www.w3.org/2000/svg' width='1024' height='1024' viewBox='0 0 1024 1024'>
  <defs>
    <linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>
      <stop offset='0%' stop-color='#09090b'/>
      <stop offset='55%' stop-color='{accent}' stop-opacity='0.75'/>
      <stop offset='100%' stop-color='#020617'/>
    </linearGradient>
    <filter id='shadow' x='-20%' y='-20%' width='140%' height='140%'>
      <feDropShadow dx='0' dy='16' stdDeviation='24' flood-color='#000000' flood-opacity='0.45'/>
    </filter>
  </defs>
  <rect width='1024' height='1024' fill='url(#bg)'/>
  <circle cx='810' cy='180' r='180' fill='white' opacity='0.10'/>
  <circle cx='210' cy='860' r='240' fill='white' opacity='0.08'/>
  <g filter='url(#shadow)'>
    <rect x='88' y='120' rx='36' ry='36' width='848' height='784' fill='rgba(10, 10, 16, 0.80)' stroke='rgba(255,255,255,0.12)'/>
    <text x='136' y='210' fill='white' font-family='Segoe UI, Arial, sans-serif' font-size='54' font-weight='700'>{html.escape(title)}</text>
    <text x='136' y='288' fill='rgba(255,255,255,0.78)' font-family='Segoe UI, Arial, sans-serif' font-size='28'>Style: {safe_style}</text>
    <text x='136' y='336' fill='rgba(255,255,255,0.78)' font-family='Segoe UI, Arial, sans-serif' font-size='28'>Aspect ratio: {safe_ratio}</text>
    <text x='136' y='430' fill='rgba(255,255,255,0.92)' font-family='Segoe UI, Arial, sans-serif' font-size='34'>Prompt</text>
    <foreignObject x='136' y='470' width='760' height='250'>
      <div xmlns='http://www.w3.org/1999/xhtml' style='color: rgba(255,255,255,0.88); font-size: 30px; line-height: 1.35; font-family: Segoe UI, Arial, sans-serif; word-wrap: break-word;'>
        {safe_prompt}
      </div>
    </foreignObject>
    <rect x='136' y='760' rx='24' ry='24' width='300' height='74' fill='{accent}'/>
    <text x='190' y='808' fill='#08111f' font-family='Segoe UI, Arial, sans-serif' font-size='28' font-weight='700'>Fooocus-ready</text>
  </g>
</svg>"""


class FooocusBridgeProvider(ImageProvider):
    provider_name: ProviderName = "fooocus"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        raise RuntimeError(
            "Il bridge Fooocus è predisposto ma non ancora collegato a un endpoint specifico. "
            "Imposta IMAGE_AI_PROVIDER=mock per usare subito la demo locale, oppure collega qui il tuo servizio immagine locale o remoto."
        )


class LocalDiffusersProvider(ImageProvider):
    provider_name: ProviderName = "local"

    def __init__(
        self,
        model_id: str,
        device: str,
        num_inference_steps: int,
        guidance_scale: float,
        enable_personal_lora: bool,
        personal_lora_path: str,
        personal_lora_scale: float,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.enable_personal_lora = enable_personal_lora
        self.personal_lora_path = personal_lora_path
        self.personal_lora_scale = personal_lora_scale
        self._pipeline: Any | None = None
        self._effective_steps = num_inference_steps
        self._active_with_personal_model = False

    def _check_available_memory(self) -> bool:
        """Check if enough memory is available for safe generation."""
        try:
            memory = psutil.virtual_memory()
            # On CPU: reserve at least 2GB free (conservative)
            min_free_bytes = 2 * 1024 * 1024 * 1024
            return memory.available >= min_free_bytes
        except Exception:
            return True  # Assume OK if psutil fails

    def _resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_personal_lora_path(self) -> Path:
        candidate = Path(self.personal_lora_path)
        if candidate.is_absolute():
            return candidate
        return (ROOT_DIR / candidate).resolve()

    def _ensure_pipeline(self, use_personal_model: bool) -> Any:
        if StableDiffusionPipeline is None or torch is None:
            raise RuntimeError(
                "Provider locale non disponibile: installa dipendenze con 'pip install diffusers transformers accelerate torch safetensors'."
            )
        if self._pipeline is not None and self._active_with_personal_model == use_personal_model:
            return self._pipeline

        target_device = self._resolve_device()
        dtype = torch.float16 if target_device == "cuda" else torch.float32
        model_source = self.model_id
        candidate = Path(self.model_id)
        if not candidate.is_absolute():
            local_candidate = (ROOT_DIR / candidate).resolve()
            if local_candidate.exists():
                model_source = str(local_candidate)
        elif candidate.exists():
            model_source = str(candidate)

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_source,
            torch_dtype=dtype,
            local_files_only=Path(model_source).exists(),
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(target_device)
        if target_device == "cpu":
            pipeline.enable_attention_slicing()

        if use_personal_model:
            lora_path = self._resolve_personal_lora_path()
            if not lora_path.exists():
                raise RuntimeError(
                    f"Modello personale non trovato: {lora_path}. "
                    "Verifica il training o aggiorna local_personal_lora_path in /api/config."
                )
            pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=self.personal_lora_scale)

        self._pipeline = pipeline
        self._active_with_personal_model = use_personal_model
        return self._pipeline

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        use_personal_model = context.use_personal_model and self.enable_personal_lora
        pipeline = self._ensure_pipeline(use_personal_model)
        requested_width, requested_height = _parse_aspect_ratio(context.aspect_ratio)
        device = self._resolve_device()

        # Keep local CPU inference inside a safe memory envelope.
        # CPU is very conservative to prevent OOM with diffusion models.
        if device == "cpu":
            max_side = 384  # Conservative: reduced from 512
            min_side = 256
            max_count = 1
            # Auto-reduce inference steps on CPU for memory safety
            self._effective_steps = min(self.num_inference_steps, 15)
        else:
            max_side = 1024
            min_side = 512
            max_count = 4
            self._effective_steps = self.num_inference_steps
            # Check available memory on GPU as well
            if not self._check_available_memory():
                max_side = 768  # Fallback to smaller resolution
                max_count = 2

        width = max(min_side, min(requested_width, max_side))
        height = max(min_side, min(requested_height, max_side))
        width = (width // 8) * 8
        height = (height // 8) * 8
        safe_count = max(1, min(context.count, max_count))

        results: list[ImageItem] = []
        for index in range(safe_count):
            seed_value = context.seed if context.seed is not None else random.randint(1, 2_147_483_647)
            seed_value = int(seed_value) + index
            generator = torch.Generator(device=device).manual_seed(seed_value)
            try:
                rendered = pipeline(
                    prompt=context.prompt,
                    negative_prompt=context.negative_prompt or None,
                    num_inference_steps=self._effective_steps,
                    guidance_scale=self.guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "not enough memory" in message or "out of memory" in message:
                    raise RuntimeError(
                        f"Memoria insufficiente per la generazione locale (device={device}). "
                        f"Sistema ha applicato limiti di sicurezza: {width}x{height}@{self._effective_steps}steps, count={safe_count}. "
                        f"Se il problema persiste, riduci ulteriormente risoluzione/aspect ratio oppure usa solo count=1."
                    ) from exc
                raise
            image = rendered.images[0]
            _save_training_sample(
                image=image,
                prompt=context.prompt,
                style=context.style,
                aspect_ratio=context.aspect_ratio,
                seed=seed_value,
                negative_prompt=context.negative_prompt,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            results.append(
                ImageItem(
                    title=f"Render {index + 1}",
                    image_data_uri=f"data:image/png;base64,{encoded}",
                    prompt=context.prompt,
                    notes=(
                        _build_preview_notes(context)
                      + f" | Render mode: local-diffusers | model: {self.model_id} | seed: {seed_value}"
                      + (f" | personal-lora: on ({self.personal_lora_scale:.2f})" if use_personal_model else " | personal-lora: off")
                      + (" | adjusted-for-cpu" if device == "cpu" else "")
                    ),
                )
            )
        return results


def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    try:
        width_str, height_str = aspect_ratio.lower().split("x", maxsplit=1)
        return int(width_str), int(height_str)
    except Exception:
        return 1024, 1024


def _apply_style_prompt(prompt: str, negative_prompt: str, style: str) -> tuple[str, str]:
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Default"])
    styled_prompt = " ".join(
        part for part in [preset.get("prefix", "").strip(), prompt.strip(), preset.get("suffix", "").strip()] if part
    )
    style_negative = preset.get("negative", "").strip()
    merged_negative = ", ".join(part for part in [negative_prompt.strip(), style_negative] if part)
    return styled_prompt or prompt, merged_negative


def _save_training_sample(
  image: Any,
  prompt: str,
  style: str,
  aspect_ratio: str,
  seed: int,
  negative_prompt: str,
) -> None:
  """Persist generated samples to gradually build a personal training dataset."""
  try:
    TRAINING_RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"sample_{timestamp}_{seed}"
    image_path = TRAINING_RAW_DIR / f"{base_name}.png"
    image.save(image_path, format="PNG", optimize=True)

    caption = f"{prompt}. style: {style}. aspect_ratio: {aspect_ratio}."
    metadata = {
      "image": image_path.name,
      "caption": caption,
      "prompt": prompt,
      "negative_prompt": negative_prompt,
      "style": style,
      "aspect_ratio": aspect_ratio,
      "seed": seed,
      "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (TRAINING_RAW_DIR / f"{base_name}.json").write_text(
      json.dumps(metadata, ensure_ascii=True, indent=2),
      encoding="utf-8",
    )
  except Exception:
    # Dataset export should never interrupt normal generation flow.
    return


def _default_profiles() -> dict[str, ProfileDefinition]:
    return {name: ProfileDefinition(**definition) for name, definition in DEFAULT_PROFILES.items()}


def load_config() -> StudioConfig:
    if CONFIG_PATH.exists():
        try:
            raw_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            profiles = raw_data.get("profiles") or {}
            normalized_profiles = {
                name: ProfileDefinition(**profile)
                for name, profile in profiles.items()
            }
            return StudioConfig(
                provider=raw_data.get("provider", "mock"),
                fooocus_endpoint=raw_data.get("fooocus_endpoint", "http://127.0.0.1:7865"),
                local_model_id=raw_data.get("local_model_id", "runwayml/stable-diffusion-v1-5"),
                local_device=raw_data.get("local_device", "auto"),
                local_num_inference_steps=raw_data.get("local_num_inference_steps", 25),
                local_guidance_scale=raw_data.get("local_guidance_scale", 7.5),
              local_use_personal_lora=raw_data.get("local_use_personal_lora", True),
              local_personal_lora_path=raw_data.get("local_personal_lora_path", str(PERSONAL_LORA_DEFAULT_PATH)),
              local_personal_lora_scale=raw_data.get("local_personal_lora_scale", 1.0),
                profiles=normalized_profiles or _default_profiles(),
            )
        except Exception:
            pass
    return StudioConfig(profiles=_default_profiles())


def save_config(config: StudioConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(config.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def create_provider(config: StudioConfig) -> ImageProvider:
  def _env_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
      return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

  provider_name = os.getenv("IMAGE_AI_PROVIDER", config.provider).strip().lower()
  if provider_name == "fooocus":
    endpoint = os.getenv("FOOOCUS_ENDPOINT", config.fooocus_endpoint)
    return FooocusBridgeProvider(endpoint)
  if provider_name == "local":
    model_id = os.getenv("LOCAL_MODEL_ID", config.local_model_id)
    device = os.getenv("LOCAL_DEVICE", config.local_device)
    lora_path = os.getenv("LOCAL_PERSONAL_LORA_PATH", config.local_personal_lora_path)
    lora_scale = float(os.getenv("LOCAL_PERSONAL_LORA_SCALE", str(config.local_personal_lora_scale)))
    use_lora = _env_to_bool(os.getenv("LOCAL_USE_PERSONAL_LORA"), config.local_use_personal_lora)
    try:
      return LocalDiffusersProvider(
        model_id=model_id,
        device=device,
        num_inference_steps=config.local_num_inference_steps,
        guidance_scale=config.local_guidance_scale,
        enable_personal_lora=use_lora,
        personal_lora_path=lora_path,
        personal_lora_scale=lora_scale,
      )
    except Exception:
      return MockImageProvider()
  return MockImageProvider()


studio_config = load_config()
provider = create_provider(studio_config)
app = FastAPI(title="Personal Image Studio", version="0.1.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _index_html()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": provider.provider_name}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    personal_model_path = Path(studio_config.local_personal_lora_path)
    if not personal_model_path.is_absolute():
      personal_model_path = (ROOT_DIR / personal_model_path).resolve()
    return {
    "providers": ["mock", "fooocus", "local"],
        "actions": ["generate", "get_config", "update_config", "list_profiles"],
    "styles": AVAILABLE_STYLES,
    "personal_model": {
      "enabled": studio_config.local_use_personal_lora,
      "path": str(personal_model_path),
      "exists": personal_model_path.exists(),
      "scale": studio_config.local_personal_lora_scale,
    },
        "config_path": str(CONFIG_PATH),
    }


@app.get("/api/config")
def get_config() -> StudioConfig:
    return studio_config


@app.put("/api/config")
def update_config(update: ConfigUpdate) -> StudioConfig:
    global studio_config, provider
    if update.provider is not None:
        studio_config.provider = update.provider
    if update.fooocus_endpoint is not None:
        studio_config.fooocus_endpoint = update.fooocus_endpoint
    if update.local_model_id is not None:
      studio_config.local_model_id = update.local_model_id
    if update.local_device is not None:
      studio_config.local_device = update.local_device
    if update.local_num_inference_steps is not None:
      studio_config.local_num_inference_steps = update.local_num_inference_steps
    if update.local_guidance_scale is not None:
      studio_config.local_guidance_scale = update.local_guidance_scale
    if update.local_use_personal_lora is not None:
      studio_config.local_use_personal_lora = update.local_use_personal_lora
    if update.local_personal_lora_path is not None:
      studio_config.local_personal_lora_path = update.local_personal_lora_path
    if update.local_personal_lora_scale is not None:
      studio_config.local_personal_lora_scale = update.local_personal_lora_scale
    if update.profiles is not None:
        studio_config.profiles = update.profiles
    if not studio_config.profiles:
        studio_config.profiles = _default_profiles()
    save_config(studio_config)
    provider = create_provider(studio_config)
    return studio_config


@app.get("/api/profiles")
def list_profiles() -> dict[str, ProfileDefinition]:
    return studio_config.profiles


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
  profile = studio_config.profiles.get(request.profile) or studio_config.profiles.get("default")
  selected_style = request.style if request.style != "Default" else (profile.style if profile else "Default")
  selected_ratio = request.aspect_ratio if request.aspect_ratio != "1024x1024" else (profile.aspect_ratio if profile else "1024x1024")
  selected_count = request.count if request.count != 1 else (profile.count if profile else 1)
  translated_prompt = _translate_to_english(request.prompt)
  translated_negative = _translate_to_english(request.negative_prompt)
  styled_prompt, styled_negative = _apply_style_prompt(translated_prompt, translated_negative, selected_style)
  context = GenerationContext(
    prompt=styled_prompt,
    negative_prompt=styled_negative,
    creative_note=request.creative_note,
    profile=request.profile,
    style=selected_style,
    aspect_ratio=selected_ratio,
    seed=request.seed,
    count=selected_count,
    use_personal_model=request.use_personal_model,
  )
  items = provider.generate(context)
  return GenerateResponse(provider=provider.provider_name, profile=request.profile, items=items)


@app.exception_handler(Exception)
def handle_unexpected_error(_: object, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def _build_preview_notes(context: GenerationContext) -> str:
    parts = [f"Profilo: {context.profile}", f"Style: {context.style}", f"Aspect ratio: {context.aspect_ratio}"]
    if context.creative_note:
        parts.append(f"Nota creativa: {context.creative_note}")
    return " | ".join(parts)


def _index_html() -> str:
    return """<!doctype html>
<html lang='it'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Personal Image Studio</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #050814;
      --panel: rgba(12, 18, 32, 0.82);
      --panel-border: rgba(255, 255, 255, 0.12);
      --text: #eef2ff;
      --muted: rgba(238, 242, 255, 0.72);
      --accent: #22c55e;
      --accent-2: #38bdf8;
      --danger: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'Segoe UI', Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(34,197,94,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(56,189,248,0.14), transparent 24%),
        linear-gradient(145deg, #02030a 0%, #060b18 48%, #09121f 100%);
      color: var(--text);
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    header {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 24px;
      align-items: end;
      margin-bottom: 22px;
    }
    .hero {
      padding: 28px;
      border: 1px solid var(--panel-border);
      border-radius: 28px;
      background: linear-gradient(180deg, rgba(20,28,48,0.88), rgba(8,12,24,0.72));
      box-shadow: 0 24px 80px rgba(0,0,0,0.34);
    }
    .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .18em; font-size: 12px; margin: 0 0 10px; }
    h1 { margin: 0; font-size: clamp(36px, 5vw, 64px); line-height: .95; }
    .lead { margin: 14px 0 0; max-width: 68ch; color: var(--muted); font-size: 17px; line-height: 1.6; }
    .status-card {
      padding: 20px 22px;
      border-radius: 24px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--panel-border);
    }
    .status-card strong { display: block; font-size: 15px; margin-bottom: 8px; }
    .status-card p { margin: 0; color: var(--muted); line-height: 1.5; }
    .mini { margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); }
    .keyvalue { display: grid; grid-template-columns: 120px 1fr; gap: 8px; margin: 8px 0; color: var(--muted); font-size: 13px; }
    .keyvalue strong { color: var(--text); font-weight: 600; }
    .grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 20px;
    }
    .panel {
      padding: 20px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      box-shadow: 0 24px 80px rgba(0,0,0,0.28);
    }
    label { display: block; font-size: 13px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin-bottom: 8px; }
    textarea, select, input {
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(4,8,18,0.86);
      color: var(--text);
      padding: 14px 15px;
      font: inherit;
      outline: none;
    }
    textarea { min-height: 158px; resize: vertical; }
    .field { margin-bottom: 16px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .ratio-picker { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 8px; }
    .actions { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    button {
      appearance: none;
      border: 0;
      border-radius: 16px;
      padding: 14px 20px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: #041019;
      background: linear-gradient(135deg, var(--accent), #86efac);
      box-shadow: 0 18px 30px rgba(34,197,94,0.24);
    }
    .ghost {
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border: 1px solid var(--panel-border);
      box-shadow: none;
    }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .gallery { display: grid; gap: 18px; }
    .card {
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid var(--panel-border);
      background: rgba(10, 14, 25, 0.94);
    }
    .card img { display: block; width: 100%; height: auto; }
    .card-body { padding: 18px 18px 20px; }
    .card-body h3 { margin: 0 0 8px; font-size: 18px; }
    .card-body p { margin: 0; color: var(--muted); line-height: 1.55; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(56,189,248,0.12);
      color: #bae6fd;
      border: 1px solid rgba(56,189,248,0.18);
      margin-bottom: 14px;
    }
    .empty {
      padding: 28px;
      color: var(--muted);
      border: 1px dashed rgba(255,255,255,0.16);
      border-radius: 24px;
      background: rgba(255,255,255,0.02);
    }
    @media (max-width: 980px) {
      header, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class='shell'>
    <header>
      <section class='hero'>
        <p class='eyebrow'>Personal AI Image Studio</p>
        <h1>Un cockpit locale per prompt, preset e immagini.</h1>
        <p class='lead'>Questo progetto è un punto di partenza pulito per la tua AI personale. È ispirato al flusso semplice di Fooocus, ma separa il nostro codice da quello del progetto originale così puoi pubblicarlo su GitHub e adattarlo senza vincoli.</p>
      </section>
      <aside class='status-card'>
        <strong>Stato corrente</strong>
        <p>Provider attivo: <span id='providerName'>mock</span><br/>Endpoint health: <span id='healthState'>in attesa</span></p>
        <div class='mini'>
          <div class='keyvalue'><strong>API</strong><span><a href='/api/capabilities' style='color:#93c5fd;'>/api/capabilities</a></span></div>
          <div class='keyvalue'><strong>Config</strong><span><a href='/api/config' style='color:#93c5fd;'>/api/config</a></span></div>
          <div class='keyvalue'><strong>Profili</strong><span><a href='/api/profiles' style='color:#93c5fd;'>/api/profiles</a></span></div>
        </div>
      </aside>
    </header>

    <main class='grid'>
      <section class='panel'>
        <div class='badge'>Motore personalizzabile via API</div>
        <div class='field'>
          <label for='prompt'>Prompt</label>
          <textarea id='prompt' placeholder='Esempio: ritratto cinematografico di un androide in una città al tramonto'></textarea>
        </div>
        <div class='field'>
          <label for='profile'>Profilo</label>
          <select id='profile'></select>
        </div>
        <div class='field'>
          <label for='negativePrompt'>Negative prompt</label>
          <textarea id='negativePrompt' placeholder='Esempio: blurry, low quality, extra fingers'></textarea>
        </div>
        <div class='row'>
          <div class='field'>
            <label for='style'>Style</label>
            <select id='style'></select>
          </div>
          <div class='field'>
            <label for='aspectRatioFamily'>Aspect ratio e pixel</label>
            <div class='ratio-picker'>
              <select id='aspectRatioFamily'>
                <option value='1:1'>1:1</option>
                <option value='4:3'>4:3</option>
                <option value='3:4'>3:4</option>
                <option value='16:9'>16:9</option>
                <option value='9:16'>9:16</option>
              </select>
              <select id='aspectRatio'></select>
            </div>
          </div>
        </div>
        <div class='row'>
          <div class='field'>
            <label for='count'>Numero immagini</label>
            <select id='count'>
              <option value='1'>1</option>
              <option value='2'>2</option>
              <option value='3'>3</option>
              <option value='4'>4</option>
            </select>
          </div>
          <div class='field'>
            <label for='seed'>Seed opzionale</label>
            <input id='seed' type='number' placeholder='Lascia vuoto per casuale' />
          </div>
        </div>
        <div class='field'>
          <label for='creativeNote'>Nota creativa per agenti esterni</label>
          <textarea id='creativeNote' placeholder='Esempio: mantieni palette fredda, enfatizza materiali lucidi, priorità a dettagli architettonici'></textarea>
        </div>
        <div class='actions'>
          <button id='generateBtn'>Genera anteprime</button>
          <button class='ghost' id='resetBtn' type='button'>Reset</button>
        </div>
        <div class='field' style='margin-top: 12px;'>
          <label for='usePersonalModel' style='display:flex;align-items:center;gap:10px;text-transform:none;letter-spacing:0;font-size:14px;'>
            <input id='usePersonalModel' type='checkbox' checked style='width:auto;accent-color:#22c55e;' />
            Usa modello personale (LoRA)
          </label>
        </div>
        <p class='hint'>Il progetto espone un contratto API aperto: tu, la UI o un agente esterno potete guidare la generazione senza toccare il nucleo dell'app.</p>
      </section>

      <section class='panel'>
        <div class='badge'>Output</div>
        <div id='output' class='gallery'>
          <div class='empty'>Nessuna immagine ancora. Scrivi un prompt e premi genera.</div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const apiBase = window.location.origin;
    const providerName = document.getElementById('providerName');
    const healthState = document.getElementById('healthState');
    const output = document.getElementById('output');
    const generateBtn = document.getElementById('generateBtn');
    const resetBtn = document.getElementById('resetBtn');
    const profileSelect = document.getElementById('profile');
    const styleSelect = document.getElementById('style');
    const aspectRatioFamilySelect = document.getElementById('aspectRatioFamily');
    const aspectRatioSelect = document.getElementById('aspectRatio');
    const usePersonalModelCheckbox = document.getElementById('usePersonalModel');

    const ratioOptions = {
      '1:1': ['1024x1024', '768x768', '512x512'],
      '4:3': ['1152x864', '1024x768', '800x600'],
      '3:4': ['864x1152', '768x1024', '600x800'],
      '16:9': ['1344x756', '1280x720', '960x540'],
      '9:16': ['756x1344', '720x1280', '540x960'],
    };

    function fillResolutionOptions(ratioFamily, preferredValue = null) {
      const options = ratioOptions[ratioFamily] || ratioOptions['1:1'];
      aspectRatioSelect.innerHTML = '';
      options.forEach((resolution) => {
        const option = document.createElement('option');
        option.value = resolution;
        option.textContent = `${ratioFamily} (${resolution} px)`;
        aspectRatioSelect.appendChild(option);
      });
      if (preferredValue && options.includes(preferredValue)) {
        aspectRatioSelect.value = preferredValue;
      }
    }

    function findRatioFamilyFromResolution(resolution) {
      for (const [ratioFamily, resolutions] of Object.entries(ratioOptions)) {
        if (resolutions.includes(resolution)) {
          return ratioFamily;
        }
      }
      return '1:1';
    }

    async function loadProfiles() {
      const response = await fetch(`${apiBase}/api/profiles`);
      if (!response.ok) {
        throw new Error(`Impossibile caricare i profili (${response.status})`);
      }
      const profiles = await response.json();
      profileSelect.innerHTML = '';
      Object.values(profiles).forEach((profile) => {
        const option = document.createElement('option');
        option.value = profile.name;
        option.textContent = `${profile.display_name} — ${profile.notes}`;
        profileSelect.appendChild(option);
      });
    }

    async function loadStyles() {
      const response = await fetch(`${apiBase}/api/capabilities`);
      if (!response.ok) {
        throw new Error(`Impossibile caricare gli stili (${response.status})`);
      }
      const data = await response.json();
      const styles = Array.isArray(data.styles) && data.styles.length ? data.styles : ['Default'];
      styleSelect.innerHTML = '';
      styles.forEach((styleName) => {
        const option = document.createElement('option');
        option.value = styleName;
        option.textContent = styleName;
        styleSelect.appendChild(option);
      });
      styleSelect.value = 'Default';
    }

    async function refreshHealth() {
      try {
        const response = await fetch(`${apiBase}/api/health`);
        if (!response.ok) {
          throw new Error(`Health check fallito (${response.status})`);
        }
        const data = await response.json();
        providerName.textContent = data.provider;
        healthState.textContent = data.status;
      } catch (error) {
        healthState.textContent = 'offline';
      }
    }

    async function loadConfig() {
      try {
        const response = await fetch(`${apiBase}/api/config`);
        if (!response.ok) {
          return;
        }
        const config = await response.json();
        if (typeof config.local_use_personal_lora === 'boolean') {
          usePersonalModelCheckbox.checked = config.local_use_personal_lora;
        }
      } catch (error) {
        // Keep default checkbox state.
      }
    }

    function renderItems(items) {
      output.innerHTML = '';
      items.forEach((item) => {
        const card = document.createElement('article');
        card.className = 'card';
        card.innerHTML = `
          <img alt='${item.title}' src='${item.image_data_uri}' />
          <div class='card-body'>
            <h3>${item.title}</h3>
            <p>${item.notes}</p>
          </div>
        `;
        output.appendChild(card);
      });
    }

    generateBtn.addEventListener('click', async () => {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) {
        alert('Inserisci un prompt.');
        return;
      }
      generateBtn.disabled = true;
      generateBtn.textContent = 'Genero...';
      try {
        const payload = {
          prompt,
          negative_prompt: document.getElementById('negativePrompt').value.trim(),
          profile: profileSelect.value,
          style: styleSelect.value,
          aspect_ratio: aspectRatioSelect.value,
          count: Number(document.getElementById('count').value),
          seed: document.getElementById('seed').value ? Number(document.getElementById('seed').value) : null,
          creative_note: document.getElementById('creativeNote').value.trim(),
          use_personal_model: usePersonalModelCheckbox.checked,
        };
        const response = await fetch(`${apiBase}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const errorText = await response.text();
          let errorMessage = 'Errore durante la generazione';
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorMessage;
          } catch (parseError) {
            if (errorText) {
              errorMessage = errorText;
            }
          }
          throw new Error(errorMessage);
        }
        const data = await response.json();
        renderItems(data.items);
        providerName.textContent = data.provider;
        healthState.textContent = 'ok';
      } catch (error) {
        output.innerHTML = `<div class='empty'>${error.message}</div>`;
      } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Genera anteprime';
      }
    });

    resetBtn.addEventListener('click', () => {
      document.getElementById('prompt').value = '';
      document.getElementById('negativePrompt').value = '';
      styleSelect.value = 'Default';
      aspectRatioFamilySelect.value = '1:1';
      fillResolutionOptions('1:1', '1024x1024');
      document.getElementById('count').value = '1';
      document.getElementById('seed').value = '';
      document.getElementById('creativeNote').value = '';
      usePersonalModelCheckbox.checked = true;
      profileSelect.value = 'default';
      output.innerHTML = "<div class='empty'>Nessuna immagine ancora. Scrivi un prompt e premi genera.</div>";
    });

    aspectRatioFamilySelect.addEventListener('change', () => {
      fillResolutionOptions(aspectRatioFamilySelect.value);
    });

    fillResolutionOptions('1:1', '1024x1024');

    loadStyles();
    loadProfiles().then(() => {
      profileSelect.value = 'default';
    });
    loadConfig();
    refreshHealth();
  </script>
</body>
</html>"""

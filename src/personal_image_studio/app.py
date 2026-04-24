from __future__ import annotations

import base64
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


ProviderName = Literal["mock", "fooocus"]

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "studio_config.json"

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


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    profile: str = Field(default="default", max_length=80)
    style: str = Field(default="Default", max_length=120)
    aspect_ratio: str = Field(default="1024x1024", max_length=20)
    count: int = Field(default=1, ge=1, le=4)
    seed: int | None = Field(default=None)
    creative_note: str = Field(default="", max_length=2000)


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
    profiles: dict[str, ProfileDefinition] = Field(default_factory=dict)


class ConfigUpdate(BaseModel):
    provider: ProviderName | None = None
    fooocus_endpoint: str | None = None
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


class ImageProvider:
    provider_name: ProviderName = "mock"

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        raise NotImplementedError


class MockImageProvider(ImageProvider):
    provider_name: ProviderName = "mock"

    def generate(self, context: GenerationContext) -> list[ImageItem]:
        items: list[ImageItem] = []
        for index in range(context.count):
            title = f"Preview {index + 1}"
            accent = self._accent_color(context.prompt, index)
            svg = self._build_svg(
                title=title,
                prompt=context.prompt,
                style=context.style,
                aspect_ratio=context.aspect_ratio,
                accent=accent,
            )
            encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            items.append(
                ImageItem(
                    title=title,
                    image_data_uri=f"data:image/svg+xml;base64,{encoded}",
                    prompt=context.prompt,
                    notes=_build_preview_notes(context),
                )
            )
        return items

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
                profiles=normalized_profiles or _default_profiles(),
            )
        except Exception:
            pass
    return StudioConfig(profiles=_default_profiles())


def save_config(config: StudioConfig) -> None:
    CONFIG_PATH.write_text(json.dumps(config.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def create_provider(config: StudioConfig) -> ImageProvider:
    provider_name = os.getenv("IMAGE_AI_PROVIDER", config.provider).strip().lower()
    if provider_name == "fooocus":
        endpoint = os.getenv("FOOOCUS_ENDPOINT", config.fooocus_endpoint)
        return FooocusBridgeProvider(endpoint)
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
    return {
        "providers": ["mock", "fooocus"],
        "actions": ["generate", "get_config", "update_config", "list_profiles"],
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
  context = GenerationContext(
    prompt=request.prompt,
    negative_prompt=request.negative_prompt,
    creative_note=request.creative_note,
    profile=request.profile,
    style=selected_style,
    aspect_ratio=selected_ratio,
    seed=request.seed,
    count=selected_count,
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
            <select id='style'>
              <option>Default</option>
              <option>Photographic</option>
              <option>Anime</option>
              <option>Cinematic</option>
              <option>Concept Art</option>
            </select>
          </div>
          <div class='field'>
            <label for='aspectRatio'>Aspect ratio</label>
            <select id='aspectRatio'>
              <option value='1024x1024'>1:1</option>
              <option value='1152x896'>4:3</option>
              <option value='896x1152'>3:4</option>
              <option value='1344x768'>16:9</option>
              <option value='768x1344'>9:16</option>
            </select>
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
          style: document.getElementById('style').value,
          aspect_ratio: document.getElementById('aspectRatio').value,
          count: Number(document.getElementById('count').value),
          seed: document.getElementById('seed').value ? Number(document.getElementById('seed').value) : null,
          creative_note: document.getElementById('creativeNote').value.trim(),
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
      document.getElementById('style').value = 'Default';
      document.getElementById('aspectRatio').value = '1024x1024';
      document.getElementById('count').value = '1';
      document.getElementById('seed').value = '';
      document.getElementById('creativeNote').value = '';
      profileSelect.value = 'default';
      output.innerHTML = "<div class='empty'>Nessuna immagine ancora. Scrivi un prompt e premi genera.</div>";
    });

    loadProfiles().then(() => {
      profileSelect.value = 'default';
    });
    refreshHealth();
  </script>
</body>
</html>"""

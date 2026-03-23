"""
Aura Chat Image Generation Worker
Supports both RunPod Serverless and Vast.ai Serverless (HTTP server mode)

Usage:
  - RunPod:  Automatically detected, uses runpod.serverless.start()
  - Vast.ai: Starts FastAPI HTTP server on port 3000
  - Direct:  WORKER_MODE=http python handler.py
"""
import os
import io
import base64
import time
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura-worker")

# ============================================================
# Global state
# ============================================================
pipe = None
loaded_character_lora = None

MODELS_DIR = os.environ.get("HF_HOME", "/workspace/models")
LORAS_DIR = "/workspace/loras"

UNCENSORED_LORA_HF = "enhanceaiteam/Flux-Uncensored-V2"

# Character LoRA URLs - hosted on Cloudflare R2 or local
CHARACTER_LORAS = {
    "luna": os.environ.get("LUNA_LORA_URL", ""),
    # Add more characters as trained
}

# ============================================================
# Model loading
# ============================================================

def download_file(url: str, dest: str):
    if os.path.exists(dest):
        logger.info(f"  [cached] {dest}")
        return
    logger.info(f"  [downloading] {url} -> {dest}")
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    logger.info(f"  [done] {os.path.getsize(dest) / 1024 / 1024:.0f}MB")


def load_pipeline():
    global pipe
    if pipe is not None:
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logger.info("=== Loading FLUX Dev pipeline ===")
    start = time.time()

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    # Sequential offload for 24GB GPUs; switch to enable_model_cpu_offload for 48GB+
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    if vram_gb >= 40:
        pipe.enable_model_cpu_offload()
        logger.info(f"  Using model_cpu_offload (VRAM: {vram_gb:.0f}GB)")
    else:
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        logger.info(f"  Using sequential_cpu_offload + VAE tiling (VRAM: {vram_gb:.0f}GB)")

    # Load uncensored LoRA
    logger.info("  Loading Uncensored LoRA...")
    pipe.load_lora_weights(UNCENSORED_LORA_HF, adapter_name="uncensored")

    elapsed = time.time() - start
    logger.info(f"=== Pipeline ready in {elapsed:.0f}s ===")


def load_character_lora(character: str):
    global pipe, loaded_character_lora

    if loaded_character_lora == character:
        return

    # Check local file first, then URL
    local_path = f"{LORAS_DIR}/{character}-character.safetensors"
    lora_url = CHARACTER_LORAS.get(character, "")

    if os.path.exists(local_path):
        logger.info(f"  Loading {character} LoRA from local: {local_path}")
        pipe.load_lora_weights(local_path, adapter_name=character)
    elif lora_url:
        dest = f"{LORAS_DIR}/{character}-character.safetensors"
        download_file(lora_url, dest)
        pipe.load_lora_weights(dest, adapter_name=character)
    else:
        # No character LoRA available, use uncensored only
        pipe.set_adapters(["uncensored"], adapter_weights=[0.8])
        loaded_character_lora = None
        return

    pipe.set_adapters(
        ["uncensored", character],
        adapter_weights=[0.6, 0.8]
    )
    loaded_character_lora = character


# ============================================================
# Generation core
# ============================================================

def generate_image(params: dict) -> dict:
    """Core generation function shared by all server modes."""
    prompt = params.get("prompt", "")
    character = params.get("character", None)
    width = params.get("width", 768)
    height = params.get("height", 768)
    steps = params.get("num_inference_steps", 25)
    guidance = params.get("guidance_scale", 3.5)
    seed = params.get("seed", None)
    uncensored_weight = params.get("uncensored_weight", 0.6)
    character_weight = params.get("character_weight", 0.8)

    if not prompt:
        return {"error": "prompt is required"}

    load_pipeline()

    if character:
        load_character_lora(character)
        # Allow dynamic weight adjustment per request
        adapters = ["uncensored", character] if loaded_character_lora == character else ["uncensored"]
        weights = [uncensored_weight, character_weight] if len(adapters) == 2 else [uncensored_weight]
        pipe.set_adapters(adapters, adapter_weights=weights)
    else:
        pipe.set_adapters(["uncensored"], adapter_weights=[uncensored_weight])

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(f"Generating: {prompt[:80]}... ({width}x{height}, {steps} steps)")
    torch.cuda.empty_cache()
    start = time.time()

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    elapsed = time.time() - start
    logger.info(f"Generated in {elapsed:.1f}s")

    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=90)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": img_base64,
        "format": "webp",
        "width": width,
        "height": height,
        "generation_time_seconds": round(elapsed, 2),
    }


# ============================================================
# Server modes
# ============================================================

def start_http_server():
    """Vast.ai / generic HTTP mode — FastAPI server on port 3000."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import Optional
    import uvicorn

    app = FastAPI(title="Aura Image Gen")

    class GenRequest(BaseModel):
        prompt: str
        character: Optional[str] = None
        width: int = 768
        height: int = 768
        num_inference_steps: int = 25
        guidance_scale: float = 3.5
        seed: Optional[int] = None
        uncensored_weight: float = 0.6
        character_weight: float = 0.8

    @app.get("/health")
    def health():
        return {"status": "ok", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}

    @app.post("/generate")
    def generate(req: GenRequest):
        return generate_image(req.model_dump())

    # Vast.ai pyworker compatibility
    @app.post("/generate/sync")
    def generate_sync(req: GenRequest):
        return generate_image(req.model_dump())

    port = int(os.environ.get("PORT", "3000"))
    logger.info(f"Starting HTTP server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def start_runpod():
    """RunPod Serverless mode."""
    import runpod

    def handler(job):
        return generate_image(job["input"])

    runpod.serverless.start({"handler": handler})


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    mode = os.environ.get("WORKER_MODE", "auto")

    if mode == "http":
        start_http_server()
    elif mode == "runpod":
        start_runpod()
    else:
        # Auto-detect: if runpod package is available and we're in RunPod env, use it
        try:
            import runpod
            if os.environ.get("RUNPOD_POD_ID"):
                start_runpod()
            else:
                start_http_server()
        except ImportError:
            start_http_server()

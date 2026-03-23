"""
Aura Chat Image Generation Worker for RunPod Serverless
FLUX Dev + Uncensored LoRA + Character LoRA
"""
import os
import io
import base64
import time
import torch
import runpod
from pathlib import Path

# Global pipeline - loaded once, reused across requests
pipe = None
loaded_character_lora = None

MODELS_DIR = "/models"
LORAS_DIR = f"{MODELS_DIR}/loras"

# LoRA URLs - hosted on HuggingFace (public)
UNCENSORED_LORA_URL = os.environ.get(
    "UNCENSORED_LORA_URL",
    "https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2/resolve/main/lora.safetensors"
)

# Character LoRA URLs - map character name to URL
CHARACTER_LORAS = {
    "luna": os.environ.get("LUNA_LORA_URL", "https://pub-4feac03f1ed6434e92e77e654d66ef68.r2.dev/models/loras/luna-character.safetensors"),
    # Add more characters as they are trained
}

def download_file(url: str, dest: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        print(f"  [cached] {dest}")
        return
    print(f"  [downloading] {url} -> {dest}")
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  [done] {os.path.getsize(dest) / 1024 / 1024:.0f}MB")

def load_pipeline():
    """Load FLUX Dev pipeline + Uncensored LoRA. Only runs on first request."""
    global pipe
    if pipe is not None:
        return

    print("=== Loading FLUX Dev pipeline ===")
    start = time.time()

    from diffusers import FluxPipeline

    # Download uncensored LoRA
    uncensored_path = f"{LORAS_DIR}/flux-uncensored-v2.safetensors"
    download_file(UNCENSORED_LORA_URL, uncensored_path)

    # Load FLUX Dev FP8 (auto-downloads from HuggingFace, ~17GB first time)
    print("  Loading FLUX Dev model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Load uncensored LoRA as base
    print("  Loading Uncensored LoRA...")
    pipe.load_lora_weights(
        uncensored_path,
        adapter_name="uncensored"
    )

    elapsed = time.time() - start
    print(f"=== Pipeline ready in {elapsed:.0f}s ===")

def load_character_lora(character: str):
    """Load a character-specific LoRA on top of the uncensored base."""
    global pipe, loaded_character_lora

    if loaded_character_lora == character:
        return  # Already loaded

    lora_url = CHARACTER_LORAS.get(character, "")
    if not lora_url:
        # No character LoRA, just use uncensored
        pipe.set_adapters(["uncensored"], adapter_weights=[0.8])
        loaded_character_lora = None
        return

    char_path = f"{LORAS_DIR}/{character}-character.safetensors"
    download_file(lora_url, char_path)

    print(f"  Loading {character} character LoRA...")
    pipe.load_lora_weights(
        char_path,
        adapter_name=character
    )

    # Stack both LoRAs: uncensored (for NSFW capability) + character (for consistency)
    pipe.set_adapters(
        ["uncensored", character],
        adapter_weights=[0.6, 0.8]  # Character LoRA stronger than uncensored
    )
    loaded_character_lora = character

def handler(job):
    """RunPod handler function - processes image generation requests."""
    job_input = job["input"]

    # Extract parameters
    prompt = job_input.get("prompt", "")
    character = job_input.get("character", None)  # Optional character name
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("num_inference_steps", 28)
    guidance = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", None)

    if not prompt:
        return {"error": "prompt is required"}

    # Load pipeline (cached after first call)
    load_pipeline()

    # Load character LoRA if specified
    if character:
        load_character_lora(character)
    else:
        pipe.set_adapters(["uncensored"], adapter_weights=[0.8])

    # Generate
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Generating: {prompt[:80]}... ({width}x{height}, {steps} steps)")
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
    print(f"Generated in {elapsed:.1f}s")

    # Convert to base64
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

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})

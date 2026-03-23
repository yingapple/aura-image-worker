"""
Vast.ai PyWorker configuration for Aura Chat Image Generation.
Proxies requests to the FastAPI model server running on port 3000.
"""
import random
import sys

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 3000
MODEL_LOG_FILE = "/workspace/server.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# Log patterns for pyworker to detect server state
MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
    "Uvicorn running on",
]

MODEL_ERROR_LOG_MSGS = [
    "Traceback (most recent call last):",
    "RuntimeError:",
    "torch.OutOfMemoryError:",
    "CUDA out of memory",
]

MODEL_INFO_LOG_MSGS = [
    "Loading FLUX Dev",
    "Loading Uncensored LoRA",
    "Loading pipeline components",
    "Generating:",
]

# Benchmark prompts for capacity estimation
benchmark_prompts = [
    "LUNACHR, anime girl with silver-lavender hair, violet eyes, casual outfit, smiling, warm lighting",
    "anime girl, elegant dress, garden scene, soft bokeh, detailed art",
    "anime girl, coffee shop, cozy atmosphere, warm colors, detailed illustration",
]

benchmark_dataset = [
    {
        "prompt": prompt,
        "width": 768,
        "height": 768,
        "num_inference_steps": 10,  # fewer steps for benchmark speed
        "guidance_scale": 3.5,
        "seed": random.randint(0, sys.maxsize),
    }
    for prompt in benchmark_prompts
]

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,
            max_queue_time=300.0,  # 5 min queue (first call downloads model)
            workload_calculator=lambda payload: 100.0,  # constant cost per image
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

Worker(worker_config).run()

# Aura Chat Image Generation Worker
# Supports RunPod Serverless and Vast.ai Serverless (HTTP mode)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models
ENV WORKER_MODE=http
ENV PORT=3000
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies (pinned versions that work with PyTorch 2.4.0)
RUN pip install --no-cache-dir \
    "diffusers==0.30.3" \
    "transformers==4.44.2" \
    "accelerate==1.0.0" \
    "safetensors==0.4.5" \
    "peft==0.12.0" \
    "sentencepiece" \
    "protobuf" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.31.0" \
    "runpod==1.7.0"

# Create directories
RUN mkdir -p /workspace/models /workspace/loras

# Copy handler
COPY handler.py /workspace/handler.py

WORKDIR /workspace
EXPOSE 3000

CMD ["python", "-u", "/workspace/handler.py"]

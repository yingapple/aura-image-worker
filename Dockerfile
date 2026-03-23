# Aura Chat Image Generation Worker
# Supports: Vast.ai Serverless (pyworker) + RunPod Serverless + HTTP standalone
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models
ENV WORKER_MODE=http
ENV PORT=3000
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
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
    "runpod==1.7.0" \
    "vastai-sdk"

# Create directories
RUN mkdir -p /workspace/models /workspace/loras

# Copy all files
COPY handler.py /workspace/handler.py
COPY worker.py /workspace/worker.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

WORKDIR /workspace
EXPOSE 3000

# Default: start both model server + pyworker
CMD ["/workspace/start.sh"]

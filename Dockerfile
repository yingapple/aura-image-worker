# Aura Chat Image Generation Worker
# Based on RunPod's CUDA base image
FROM runpod/base:0.6.2-cuda12.2.0

# Set environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    diffusers==0.31.0 \
    transformers==4.46.0 \
    accelerate==1.1.0 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    protobuf==5.28.0 \
    peft==0.13.0 \
    runpod==1.7.0

# Create model directories
RUN mkdir -p /models/loras /models/huggingface

# Copy handler
COPY handler.py /handler.py

# RunPod entry point
CMD ["python", "-u", "/handler.py"]

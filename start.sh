#!/bin/bash
# Startup script: launches model server + pyworker
set -e

echo "=== Starting Aura Image Generation Server ==="

# Start the FastAPI model server in background
export WORKER_MODE=http
export PORT=3000
python3 -u /workspace/handler.py > /workspace/server.log 2>&1 &
MODEL_PID=$!
echo "Model server started (PID: $MODEL_PID)"

# Wait for model server to be ready
echo "Waiting for model server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:3000/health > /dev/null 2>&1; then
        echo "Model server ready!"
        break
    fi
    sleep 2
done

# Start the pyworker (blocks, handles autoscaling)
echo "Starting PyWorker..."
python3 -u /workspace/worker.py

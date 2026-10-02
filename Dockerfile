# Demo image for the people-analytics Streamlit dashboard (CPU by default).
# For GPU, base on `nvidia/cuda:12.1.0-runtime-ubuntu22.04` + install CUDA torch,
# and run with `--gpus all`.
FROM python:3.11-slim

# System libraries OpenCV / video I/O need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[detect,track,app]"

# Clips live outside the image; upload via the dashboard or mount a volume:
#   docker run -p 8501:8501 -v "$PWD/data:/app/data" people-analytics
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/people_analytics/app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# PyTorch نسخة CPU (أخف بكثير)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir iopaint "rembg[cpu]" python-multipart

ENV DATA_DIR=/data \
    U2NET_HOME=/data/models \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache

WORKDIR /app
COPY app.py index.html ./
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

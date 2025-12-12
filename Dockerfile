FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Supaya install tidak nanya timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}


# Install Python 3.11 dan tools penting
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    gcc \
    gfortran \
    liblapack-dev \
    libatlas-base-dev \
    libblas-dev \
    libffi-dev \
    libsndfile1 \
    curl \
    gnupg \
    lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get purge -y software-properties-common && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# Set working directory
WORKDIR /app

# Copy source code setelah dependency (optimalkan cache layer)
COPY . .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Gunakan Poetry atau pip, sesuaikan dengan environment kamu:
# Contoh dengan pip:
RUN python -m pip install .
RUN python ./demo/realtime_model_inference_from_file.py
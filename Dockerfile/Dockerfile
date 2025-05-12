# ベースイメージ
FROM python:3.9-slim

# 作業ディレクトリの作成
WORKDIR /app

# 必要なシステムライブラリのインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# 必要なライブラリのインストール
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install opencv-python jupyterlab matplotlib scikit-image

# JupyterLabを起動するための設定
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

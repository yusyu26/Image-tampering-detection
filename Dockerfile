FROM python:3.6-slim

# システムライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# pip を TF1.x 対応に固定
RUN pip install --upgrade pip==20.3.4

# Python パッケージ（バージョン指定でビルド高速化）
RUN pip install \
    tensorflow==1.13.1 \
    tflearn==0.3.2 \
    opencv-python==4.1.2.30 \
    scikit-image==0.16.2 \
    matplotlib==3.1.3 \
    pandas==1.0.1 \
    tqdm==4.41.1 \
    jupyterlab==2.0.1

WORKDIR /app
COPY . /app

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

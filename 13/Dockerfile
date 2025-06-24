FROM ubuntu:22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:1

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    nano \
    xvfb \
    x11vnc \
    fluxbox \
    xterm \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libxrandr2 \
    libxrandr-dev \
    libxss1 \
    libxss-dev \
    libxcursor1 \
    libxcursor-dev \
    libxcomposite1 \
    libxcomposite-dev \
    libxdamage1 \
    libxdamage-dev \
    libxfixes3 \
    libxfixes-dev \
    libxi6 \
    libxi-dev \
    libxrender1 \
    libxrender-dev \
    libxtst6 \
    libxtst-dev \
    libxrandr2 \
    libxrandr-dev \
    libasound2 \
    libasound2-dev \
    libpulse0 \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

# Minicondaのインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Genesisのインストール
RUN conda install -c conda-forge genesis -y

# 作業ディレクトリの設定
WORKDIR /workspace

# 必要なファイルをコピー
COPY scene.py /workspace/
COPY scene.sh /workspace/
RUN chmod +x /workspace/scene.sh

# VNCサーバーの設定
RUN mkdir -p ~/.vnc && \
    echo "sakai0314" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

# 起動スクリプトの作成
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# VNCサーバーを起動\n\
Xvfb :1 -screen 0 1280x720x16 &\n\
sleep 2\n\
\n\
# VNCサーバーを起動\n\
x11vnc -display :1 -nopw -listen localhost -xkb -ncache 10 -ncache_cr -forever &\n\
sleep 2\n\
\n\
# デスクトップ環境を起動\n\
fluxbox &\n\
sleep 2\n\
\n\
# Genesisを実行\n\
cd /workspace\n\
bash scene.sh\n\
\n\
# コンテナを維持\n\
tail -f /dev/null' > /start.sh && chmod +x /start.sh

EXPOSE 5900

CMD ["/start.sh"] 
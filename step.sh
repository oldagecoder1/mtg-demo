#!/bin/bash

# 1. Directory Setup
echo "📁 Setting up directories..."
mkdir -p /root/demo
cd /root/demo

# 2. Git Clone (HTTPS use kiya hai, no SSH setup needed)
if [ ! -d "mtg-demo" ]; then
    echo "⬇️ Cloning repository..."
    git clone https://github.com/oldagecoder1/mtg-demo.git
else
    echo "🔄 Repo already exists, pulling latest changes..."
    cd mtg-demo && git pull && cd ..
fi

# 3. Virtual Environment setup (In /tmp for SSD speed)
echo "🍏 Creating Virtual Environment in /tmp for speed..."
python3 -m venv /tmp/venv
source /tmp/venv/bin/activate

# 4. PaddlePaddle GPU & Requirements Install
echo "📦 Installing PaddlePaddle-GPU and requirements..."
pip install --upgrade pip

# PaddlePaddle Install
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Requirements Install
cd mtg-demo
pip install -r requirements_short.txt

# 5. Run the app
echo "🚀 Starting Uvicorn Server..."
uvicorn app_pcm_master_parser:app --host 0.0.0.0 --port 8000 --reload
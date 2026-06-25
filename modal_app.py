import os
import sys

import modal

app = modal.App("pcm-master-parser")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install_from_requirements("requirements_short.txt")
    .add_local_file(
        "app_pcm_master_parser.py",
        remote_path="/root/project/app_pcm_master_parser.py",
    )
    .add_local_file(
        "requirements_short.txt",
        remote_path="/root/project/requirements_short.txt",
    )
    .add_local_dir(
        "static",
        remote_path="/root/project/static",
    )
)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=16384,  # 16 GB
    timeout=3600,
)
@modal.asgi_app()
def fastapi_app():
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    from app_pcm_master_parser import app

    return app
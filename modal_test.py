import modal

app = modal.App("gpu-test")

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
    .add_local_file("test_gpu.py", remote_path="/root/project/test_gpu.py")
    .add_local_file("chemistry_short2.jpg", remote_path="/root/project/chemistry_short2.jpg")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
)

def run_test():
    import os
    import subprocess

    os.chdir("/root/project")

    subprocess.run(
        ["python", "-u", "test_gpu.py"],
        check=True,
    )
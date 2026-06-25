import time
import paddle
from paddleocr import PaddleOCRVL

print("Loading pipeline...", flush=True)

pipeline = PaddleOCRVL(
    device="gpu:0",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_layout_detection=True,
)

print("Pipeline Loaded", flush=True)

start = time.time()

output = pipeline.predict("chemistry_short2.jpg")

print("predict() finished", flush=True)

print("Type:", type(output), flush=True)

try:
    print("Length:", len(output), flush=True)
except Exception as e:
    print("Cannot get length:", e, flush=True)

print(f"Time: {time.time() - start:.2f} sec", flush=True)
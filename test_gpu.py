import time
import paddle
from paddleocr import PaddleOCRVL

print("Loading pipeline...", flush=True)

pipeline = PaddleOCRVL(
    device="gpu",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_layout_detection=True
)

print("Pipeline Loaded", flush=True)

start = time.time()

generator = pipeline.predict("chemistry_short2.jpg")

print("predict() returned generator", flush=True)

output = next(generator)

print("Got first result", flush=True)

print(f"Prediction completed in {time.time()-start:.2f} sec", flush=True)
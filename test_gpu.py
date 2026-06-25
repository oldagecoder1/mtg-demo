import time
import paddle
from paddleocr import PaddleOCRVL

print("Loading pipeline...")

pipeline = PaddleOCRVL(
    device="gpu",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_layout_detection=True
)

print("Pipeline Loaded")

start = time.time()

output = pipeline.predict("chemistry_short2.jpg")

print(f"Prediction completed in {time.time()-start:.2f} sec")

pages = list(output)

print("Pages:", len(pages))
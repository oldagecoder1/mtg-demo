from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import home, digital, handwritten
import os

app = FastAPI(title="MTG Parsers Wrapper")

# Include routers
app.include_router(home.router)
app.include_router(digital.router)
app.include_router(handwritten.router)

# Ensure output directories exist for static file mounting
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files needed by the sub-applications
app.mount("/static_outputs", StaticFiles(directory="output"), name="static_outputs")
app.mount("/static_hw", StaticFiles(directory="output"), name="static_hw")
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image
from PIL import Image
import io
import subprocess
import gdown
import os
import sys
import datetime

# Uruchomienie download_model.py na starcie
print("Sprawdzanie i pobieranie modelu...")
try:
    subprocess.run([sys.executable, "download_model.py"], check=True)
    print("Model gotowy. Uruchamianie serwera...")
except subprocess.CalledProcessError as e:
    print(f"Błąd podczas uruchamiania download_model.py: {e}")
    exit(1)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint sprawdzający status API
@app.get("/status")
async def check_status():
    return {
        "status": "online",
        "message": "API is running",
        "timestamp": datetime.datetime.now().isoformat()
    }

# Endpoint do przewidywania
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image)
    return {"brand": result["brand"], "confidence": result["confidence"]}
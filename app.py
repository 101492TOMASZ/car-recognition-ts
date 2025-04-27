from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from predict import predict_image
from PIL import Image
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dopisz swój frontendowy IP jeśli chcesz bardziej restrykcyjnie
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the response to include both the brand and confidence percentage
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image)
    return {"brand": result["brand"], "confidence": result["confidence"]}
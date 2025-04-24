from fastapi import FastAPI, File, UploadFile
from predict import predict_image
from PIL import Image
import io

app = FastAPI()

# Update the response to include both the brand and confidence percentage
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image)
    return {"brand": result["brand"], "confidence": result["confidence"]}

# api/main.py  (relevant parts)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

app = FastAPI()

MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/1")  # adjust if needed
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    return image

def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        # TFSMLayer wraps a TF SavedModel for inference in Keras 3
        tfsm_layer = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
        # Put it into a small Sequential wrapper so you can call .predict(...) normally
        MODEL = keras.Sequential([tfsm_layer])
        print("Loaded model via TFSMLayer from:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model via TFSMLayer:", e)
        MODEL = None
    return MODEL

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    # read and preprocess
    image = read_file_as_image(await file.read())
    # optionally resize to the model's expected size:
    # image = np.array(Image.fromarray(image).resize((224,224)))
    img_batch = np.expand_dims(image, 0).astype("float32")  # convert dtype if necessary
    # If your SavedModel expects normalized inputs, normalize here (e.g. /255.0)
    # img_batch = img_batch / 255.0
    preds = model.predict(img_batch)
    predicted_class = CLASS_NAMES[int(np.argmax(preds[0]))]
    confidence = float(np.max(preds[0]))
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")

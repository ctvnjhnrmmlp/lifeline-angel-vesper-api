import cv2 as cv
import numpy as np
import uvicorn
from fastapi import (FastAPI, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from PIL import Image
from fastai.vision.all import *

app = FastAPI()

image_model = load_learner("./vesper/vesper.pkl")
class_names = image_model.dls.vocab

def predict_image(image):
    # Predict the class of an image
    pred, pred_idx, probs = image_model.predict(image)
    return pred, probs[pred_idx].item()

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv.imdecode(np_img, cv.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    prediction, confidence = predict_image(pil_image)

    return {"prediction": prediction, "confidence": confidence}

@app.websocket("/api/ws/classify")
async def websocket_classify_image(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_img = np.frombuffer(data, np.uint8)
            image = cv.imdecode(np_img, cv.IMREAD_COLOR)
            if image is None:
                await websocket.send_text("Invalid image file")
                continue

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)

            prediction, confidence = predict_image(pil_image)
            await websocket.send_text(f"{prediction} with confidence {confidence:.4f}")
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

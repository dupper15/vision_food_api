import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fastapi import FastAPI, Query
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import urllib.request
import joblib

app = FastAPI()

# Load model .h5
model = joblib.load("model.pkl")  # đổi tên file nếu cần

# Mapping label nếu có (ví dụ)
label_map = {
    0: "Bánh mì",
    1: "Bánh cuốn",
    2: "Cơm tấm",
    3: "Phở",
    4: "Xôi xéo"
}

# Hàm tải ảnh từ URL
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.get("/predict")
def predict_image(url: str = Query(...)):
    try:
        print(f"🔍 API called with URL: {url}")
        image = url_to_image(url)
        image = cv2.resize(image, (224, 224))  
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_label = label_map.get(predicted_index, f"Class {predicted_index}")

        return {
            "prediction_index": predicted_index,
            "prediction_label": predicted_label
        }
    except Exception as e:
        return {"error": str(e)}

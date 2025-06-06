import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, Query
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import urllib.request
import joblib

app = FastAPI()

# Load mô hình scikit-learn
model = joblib.load("model.pkl")

# Load ResNet50 để trích đặc trưng
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

# Mapping label
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

# Hàm trích đặc trưng ảnh
def extract_features(image):
    image = cv2.resize(image, (224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = resnet_model.predict(img_array, verbose=0)
    return features  # shape (1, 2048)

@app.get("/predict")
def predict_image(url: str = Query(...)):
    try:
        print(f"🔍 API called with URL: {url}")
        image = url_to_image(url)
        features = extract_features(image)

        predicted_index = int(model.predict(features)[0])
        predicted_label = label_map.get(predicted_index, f"Class {predicted_index}")
        
       result = {
            "prediction_index": predicted_index,
            "prediction_label": predicted_label
        }

        print(f"✅ Prediction result: {result}") 
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

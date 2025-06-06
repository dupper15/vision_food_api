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

# Load ResNet50 không có top và không pooling để lấy output shape (1, 7, 7, 2048)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling=None)

# Mapping label
label_map = {
    0: "Bánh mì",
    1: "Bánh cuốn",
    2: "Cơm tấm",
    3: "Phở",
    4: "Xôi xéo"
}

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def extract_features(image):
    image = cv2.resize(image, (224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = resnet_model.predict(img_array, verbose=0)  # shape (1, 7, 7, 2048)
    features = features.reshape((features.shape[0], -1))  # flatten to (1, 100352)
    return features

@app.get("/predict")
def predict_image(url: str = Query(...)):
    try:
        print(f"🔍 API called with URL: {url}")
        image = url_to_image(url)
        features = extract_features(image)

        predicted_label = model.predict(features)[0]
        predicted_index = next((k for k, v in label_map.items() if v == predicted_label), -1)

        result = {
            "prediction_index": predicted_index,
            "prediction_label": predicted_label
        }

        print(f"✅ Prediction result: {result}") 
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

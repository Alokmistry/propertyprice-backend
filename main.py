from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import requests
import os

# ---------------- APP ----------------
app = FastAPI(title="Property Price Prediction API")
#  --------middleware ---------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ allow frontend from anywhere (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- HUGGING FACE URLS ----------------
MODEL_URL = "https://huggingface.co/Alok30m/mumbai-property-price-model/resolve/main/real_estate_model.pkl"
ENCODER_URL = "https://huggingface.co/Alok30m/mumbai-property-price-model/resolve/main/label_encoder.pkl"
FEATURES_URL = "https://huggingface.co/Alok30m/mumbai-property-price-model/resolve/main/model_features.txt"

MODEL_PATH = "real_estate_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
FEATURES_PATH = "model_features.txt"

# ---------------- DOWNLOAD UTILITY ----------------
def download_file(url: str, filename: str):
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}")
        with open(filename, "wb") as f:
            f.write(response.content)


model = None
label_encoders = None
MODEL_FEATURES = None

@app.on_event("startup")
def load_artifacts():
    global model, label_encoders, MODEL_FEATURES

    download_file(MODEL_URL, MODEL_PATH)
    download_file(ENCODER_URL, ENCODER_PATH)
    download_file(FEATURES_URL, FEATURES_PATH)

    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)

    with open(FEATURES_PATH, "r") as f:
        MODEL_FEATURES = [line.strip() for line in f.readlines()]

# ---------------- CATEGORICAL FEATURES ----------------
CATEGORICAL_FEATURES = [
    "type_of_property",
    "city",
    "area_name",
    "furnished_type"
]


# ---------------- INPUT SCHEMA ----------------
class PropertyInput(BaseModel):
    carpet_area: float
    covered_area: float
    sqft_price: float
    bedroom: int
    bathroom: int
    balconies: int
    floor_no: int
    floors: int
    type_of_property: str
    commercial: bool
    luxury_flat: bool
    city: str
    area_name: str
    isprimelocationproperty: bool
    furnished_type: str
    parking: bool
    rera: bool


# ---------------- FEATURE PREPARATION ----------------
def prepare_features(data: PropertyInput):
    input_dict = data.dict()
    feature_vector = []

    for feature in MODEL_FEATURES:
        value = input_dict.get(feature, 0)

        # Encode categorical values
        if feature in CATEGORICAL_FEATURES:
            try:
                value = label_encoders[feature].transform([value])[0]
            except Exception:
                # unseen category fallback
                value = 0

        # Convert booleans to integers
        if isinstance(value, bool):
            value = int(value)

        feature_vector.append(value)

    return np.array(feature_vector, dtype=float).reshape(1, -1)

@app.get("/")
def health():
    return {"status": "ok"}


# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/predict")
def predict_price(data: PropertyInput):
    try:
        if model is None or MODEL_FEATURES is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        X = prepare_features(data)

        prediction = model.predict(X)[0]

        return {
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

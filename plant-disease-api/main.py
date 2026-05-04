import io
import os
import torch
import requests
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Hops Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sınıf isimleri — ImageFolder'ın alfabetik sırası ile eşleşiyor:
# Disease-Downy(0), Disease-Powdery(1), Healthy(2), Insect-Pest(3)
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

CLASS_INFO = {
    "Disease-Downy": {
        "label": "Downy Mildew (Tüylü Küf)",
        "type": "disease",
        "severity": "high",
        "description": "Yapraklarda sarı-yeşil lekeler ve beyazımsı küf görülür.",
        "recommendation": "Fungisit uygulaması yapın ve etkilenen yaprakları uzaklaştırın."
    },
    "Disease-Powdery": {
        "label": "Powdery Mildew (Külleme)",
        "type": "disease",
        "severity": "medium",
        "description": "Yaprak yüzeyinde beyaz pudra görünümünde mantar.",
        "recommendation": "Sülfür bazlı fungisit uygulayın, hava sirkülasyonunu artırın."
    },
    "Healthy": {
        "label": "Sağlıklı",
        "type": "healthy",
        "severity": "none",
        "description": "Yaprak sağlıklı görünüyor.",
        "recommendation": "Rutin bakıma devam edin."
    },
    "Insect-Pest": {
        "label": "Böcek / Zararlı",
        "type": "pest",
        "severity": "medium",
        "description": "Yaprakta böcek veya zararlı böcek tespit edildi.",
        "recommendation": "Pestisit uygulayın ve biyolojik kontrol yöntemlerini değerlendirin."
    }
}

# Preprocessing - EfficientNet için standart
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model yükleme
MODEL_PATH = os.environ.get("MODEL_PATH", "efficientnet_plant.pt")
model = None

def load_model():
    global model
    try:
        model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()
        print(f"✅ Model yüklendi: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        raise

def run_inference(image: Image.Image) -> dict:
    """Ortak tahmin fonksiyonu"""
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    info = CLASS_INFO[predicted_class]
    return {
        "prediction": predicted_class,
        "label": info["label"],
        "type": info["type"],
        "severity": info["severity"],
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "all_probabilities": all_probs
    }

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"status": "ok", "message": "Hops Disease Detection API çalışıyor"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# ── Endpoint 1: Dosya yükleme (mobil uygulama gerçek kullanım) ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Görsel okunamadı")

    try:
        return run_inference(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

# ── Endpoint 2: URL'den görsel (FlutterFlow test için) ──
class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/predict-url")
async def predict_url(body: ImageURLRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        response = requests.get(body.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel indirilemedi: {str(e)}")

    try:
        return run_inference(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

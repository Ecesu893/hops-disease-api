import io
import os
import torch
import requests
import random
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import get_db, init_db, ScanHistory, User
from auth import router as auth_router, get_current_user

app = FastAPI(title="Hops Disease Detection API", version="1.0.0")

# 1. CORS Ayarları (FlutterFlow Web ve Mobil için tam izin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 2. Auth ve History Router'ı (/api prefix'i ile bağlandı)
app.include_router(auth_router, prefix="/api")

# Sınıf isimleri
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

CLASS_INFO = {
    "Disease-Downy": {
        "label": "Downy Mildew (Tüylü Küf)",
        "type": "disease",
        "severity": "high",
        "descriptions": [
            "Yaprak yüzeyinde sarı-yeşil lekeler ve alt bölgelerde beyazımsı mantar oluşumu gözlemlendi.",
            "Nemli koşullarda gelişen tüylü küf belirtileri tespit edildi.",
            "Yaprak dokusunda renk değişimi ve küf tabakası oluşumu mevcut.",
            "Bitkide Downy Mildew kaynaklı fungal enfeksiyon belirtileri görüldü.",
            "Yaprak altlarında beyaz mantar sporları ve üst yüzeyde sararmalar belirlendi."
        ],
        "recommendations": [
            "Etkilenen yaprakları budayın ve uygun fungisit uygulayın.",
            "Sulama sıklığını azaltarak hava dolaşımını artırın.",
            "Bakır içerikli veya sistemik fungisit kullanılması önerilir.",
            "Hastalığın yayılmasını önlemek için enfekte bölgeleri uzaklaştırın.",
            "Nem oranını kontrol altında tutarak koruyucu ilaçlama yapın."
        ]
    },
    "Disease-Powdery": {
        "label": "Powdery Mildew (Külleme)",
        "type": "disease",
        "severity": "medium",
        "descriptions": [
            "Yaprak yüzeyinde beyaz pudramsı mantar tabakası tespit edildi.",
            "Külleme hastalığına ait tipik beyaz fungal oluşumlar gözlemlendi.",
            "Bitki üzerinde mantar kaynaklı yüzeysel beyaz lekeler mevcut.",
            "Yapraklarda un serpilmiş görünüm oluşturan mantar enfeksiyonu bulundu.",
            "Powdery Mildew belirtileri yaprak yüzeyinde yayılmaya başlamış."
        ],
        "recommendations": [
            "Sülfür bazlı fungisit uygulaması önerilir.",
            "Bitkiler arasındaki hava akışını artırın ve aşırı nemden kaçının.",
            "Enfekte yaprakları temizleyerek düzenli kontrol sağlayın.",
            "Doğal veya kimyasal mantar önleyici ürünler kullanılabilir.",
            "Enfeksiyonun yayılmasını önlemek için sabah saatlerinde sulama yapın."
        ]
    },
    "Healthy": {
        "label": "Sağlıklı",
        "type": "healthy",
        "severity": "none",
        "descriptions": [
            "Yaprak yüzeyi sağlıklı ve doğal görünümde.",
            "Herhangi bir hastalık veya zararlı belirtisi tespit edilmedi.",
            "Bitki genel olarak sağlıklı gelişim göstermektedir.",
            "Yaprak dokusunda anormal renk değişimi veya enfeksiyon bulunmuyor."
        ],
        "recommendations": [
            "Düzenli bakım ve sulama rutinine devam edin.",
            "Periyodik gözlem yaparak bitki sağlığını koruyun.",
            "Dengeli gübreleme ve uygun ışık koşulları sağlamaya devam edin."
        ]
    },
    "Insect-Pest": {
        "label": "Böcek / Zararlı",
        "type": "pest",
        "severity": "medium",
        "descriptions": [
            "Yaprak üzerinde zararlı böcek aktivitesi tespit edildi.",
            "Bitki dokusunda böcek kaynaklı hasar belirtileri gözlemlendi.",
            "Yaprak yüzeyinde zararlı organizma izleri mevcut."
        ],
        "recommendations": [
            "Uygun pestisit veya biyolojik mücadele yöntemleri uygulayın.",
            "Zararlı yoğunluğunu azaltmak için enfekte bölgeleri temizleyin.",
            "Neem yağı veya uygun insektisit kullanımı değerlendirilebilir."
        ]
    }
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
        "prediction_class": predicted_class, # FlutterFlow uyumu için
        "label": info["label"],
        "type": info["type"],
        "severity": info["severity"],
        "confidence": round(confidence, 4),
        "confidence_score": round(confidence, 4), # FlutterFlow uyumu için
        "confidencepercent": round(confidence * 100, 1),
        "description": random.choice(info["descriptions"]),
        "recommendation": random.choice(info["recommendations"]),
        "all_probabilities": all_probs
    }

@app.on_event("startup")
async def startup_event():
    load_model()
    init_db()

@app.get("/")
def root():
    return {"status": "ok", "message": "Hops Disease Detection API çalışıyor"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Görsel okunamadı")

    try:
        result = run_inference(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

    record = ScanHistory(
        user_id=current_user.id,
        prediction_class=result["prediction"],
        confidence_score=result["confidence"],
        image_url=None,
    )
    db.add(record)
    db.commit()

    return result

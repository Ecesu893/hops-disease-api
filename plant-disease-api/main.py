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

from models import get_db, init_db, ScanHistory
from auth import router as auth_router, get_current_user
from models import User

app = FastAPI(title="Hops Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth + history endpoint'lerini bağla
app.include_router(auth_router, prefix="/api")

# Sınıf isimleri — ImageFolder'ın alfabetik sırası ile eşleşiyor:
# Disease-Downy(0), Disease-Powdery(1), Healthy(2), Insect-Pest(3)
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]


CLASS_INFO = {
    "Disease-Downy": {
        "label": "Downy Mildew (Tüylü Küf)",
        "type": "disease",
        "severity": "high",
        "description": random.choice([
            "Yaprak yüzeyinde sarı-yeşil lekeler ve alt bölgelerde beyazımsı mantar oluşumu gözlemlendi.",
            "Nemli koşullarda gelişen tüylü küf belirtileri tespit edildi.",
            "Yaprak dokusunda renk değişimi ve küf tabakası oluşumu mevcut.",
            "Bitkide Downy Mildew kaynaklı fungal enfeksiyon belirtileri görüldü.",
            "Yaprak altlarında beyaz mantar sporları ve üst yüzeyde sararmalar belirlendi."
        ]),
        "recommendation": random.choice([
            "Etkilenen yaprakları budayın ve uygun fungisit uygulayın.",
            "Sulama sıklığını azaltarak hava dolaşımını artırın.",
            "Bakır içerikli veya sistemik fungisit kullanılması önerilir.",
            "Hastalığın yayılmasını önlemek için enfekte bölgeleri uzaklaştırın.",
            "Nem oranını kontrol altında tutarak koruyucu ilaçlama yapın."
        ])
    },

    "Disease-Powdery": {
        "label": "Powdery Mildew (Külleme)",
        "type": "disease",
        "severity": "medium",
        "description": random.choice([
            "Yaprak yüzeyinde beyaz pudramsı mantar tabakası tespit edildi.",
            "Külleme hastalığına ait tipik beyaz fungal oluşumlar gözlemlendi.",
            "Bitki üzerinde mantar kaynaklı yüzeysel beyaz lekeler mevcut.",
            "Yapraklarda un serpilmiş görünüm oluşturan mantar enfeksiyonu bulundu.",
            "Powdery Mildew belirtileri yaprak yüzeyinde yayılmaya başlamış."
        ]),
        "recommendation": random.choice([
            "Sülfür bazlı fungisit uygulaması önerilir.",
            "Bitkiler arasındaki hava akışını artırın ve aşırı nemden kaçının.",
            "Enfekte yaprakları temizleyerek düzenli kontrol sağlayın.",
            "Doğal veya kimyasal mantar önleyici ürünler kullanılabilir.",
            "Sabah saatlerinde kontrollü sulama yaparak mantar gelişimini azaltın."
        ])
    },

    "Healthy": {
        "label": "Sağlıklı",
        "type": "healthy",
        "severity": "none",
        "description": random.choice([
            "Yaprak yüzeyi sağlıklı ve doğal görünümde.",
            "Herhangi bir hastalık veya zararlı belirtisi tespit edilmedi.",
            "Bitki genel olarak sağlıklı gelişim göstermektedir.",
            "Yaprak dokusunda anormal renk değişimi veya enfeksiyon bulunmuyor.",
            "Bitki sağlığı açısından olumsuz bir bulgu gözlemlenmedi."
        ]),
        "recommendation": random.choice([
            "Düzenli bakım ve sulama rutinine devam edin.",
            "Periyodik gözlem yaparak bitki sağlığını koruyun.",
            "Dengeli gübreleme ve uygun ışık koşulları sağlamaya devam edin.",
            "Koruyucu bakım uygulamaları ile bitkiyi destekleyin.",
            "Bitkinin mevcut bakım düzeni korunabilir."
        ])
    },

    "Insect-Pest": {
        "label": "Böcek / Zararlı",
        "type": "pest",
        "severity": "medium",
        "description": random.choice([
            "Yaprak üzerinde zararlı böcek aktivitesi tespit edildi.",
            "Bitki dokusunda böcek kaynaklı hasar belirtileri gözlemlendi.",
            "Yaprak yüzeyinde zararlı organizma izleri mevcut.",
            "Böcek veya larva kaynaklı deformasyon belirtileri bulundu.",
            "Bitkide pest kaynaklı enfestasyon şüphesi oluştu."
        ]),
        "recommendation": random.choice([
            "Uygun pestisit veya biyolojik mücadele yöntemleri uygulayın.",
            "Zararlı yoğunluğunu azaltmak için enfekte bölgeleri temizleyin.",
            "Doğal düşman böceklerden yararlanarak biyolojik kontrol sağlayın.",
            "Bitkiyi düzenli kontrol ederek yayılımı önleyin.",
            "Neem yağı veya uygun insektisit kullanımı değerlendirilebilir."
        ])
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
        "confidencepercent": round(confidence * 100, 1),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "all_probabilities": all_probs
    }

@app.on_event("startup")
async def startup_event():
    load_model()
    init_db()  # tabloları oluştur (users, scan_history)

@app.get("/")
def root():
    return {"status": "ok", "message": "Hops Disease Detection API çalışıyor"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# ── Endpoint 1: Dosya yükleme (mobil uygulama gerçek kullanım) ──
# Artık giriş yapılmış kullanıcı gerektiriyor (Depends(get_current_user))
# ve tahmin sonucu otomatik olarak scan_history tablosuna kaydediliyor.
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

    # Sonucu kullanıcının geçmişine kaydet
    record = ScanHistory(
        user_id=current_user.id,
        prediction_class=result["prediction"],
        confidence_score=result["confidence"],
        image_url=None,  # dosya yükleme akışında görsel URL'i yok; istersen ayrıca bir storage'a yükleyip buraya URL geçebilirsin
    )
    db.add(record)
    db.commit()

    return result

# ── Endpoint 2: URL'den görsel (FlutterFlow test için) ──
class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/predict-url")
async def predict_url(
    body: ImageURLRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        response = requests.get(body.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel indirilemedi: {str(e)}")

    try:
        result = run_inference(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

    record = ScanHistory(
        user_id=current_user.id,
        prediction_class=result["prediction"],
        confidence_score=result["confidence"],
        image_url=body.image_url,
    )
    db.add(record)
    db.commit()

    return result

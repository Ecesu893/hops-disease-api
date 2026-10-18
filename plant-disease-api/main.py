import io
import os
import uuid
import torch
import requests
import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from models import engine, get_db, init_db, ScanHistory, User
from auth import router as auth_router, get_current_user

app = FastAPI(title="Hops Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Yüklenen fotoğrafların kaydedileceği klasörü oluştur ve dışarıya aç
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Auth + history endpoint'lerini başlat
app.include_router(auth_router, prefix="/api")

# Sınıf isimleri
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
            "Yaprak altlarında beyaz mantar sporları ve üst yüzde sararmalar belirlendi."
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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model Yükleme (Lazy Loading)
MODEL_PATH = os.environ.get("MODEL_PATH", "efficientnet_plant.pt")
model = None

def get_model():
    global model
    if model is None:
        try:
            print("Model yükleniyor...")
            model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.eval()
            print(f"Model başarıyla yüklendi: {MODEL_PATH}")
        except Exception as e:
            print(f"Model yüklenemedi: {e}")
            raise HTTPException(status_code=500, detail=f"Model yüklenemedi: {str(e)}")
    return model

# Görüntü Ön İşleme (Pillow ve OpenCV ile İyileştirme)
def enhance_and_preprocess_image(image_bytes: bytes) -> Image.Image:
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Kontrast ve Parlaklık Dengeleme
        enhancer_contrast = ImageEnhance.Contrast(input_image)
        processed_img = enhancer_contrast.enhance(1.2)

        enhancer_brightness = ImageEnhance.Brightness(processed_img)
        processed_img = enhancer_brightness.enhance(1.05)

        # OpenCV ile Gürültü Azaltma (Denoise)
        open_cv_image = np.array(processed_img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        open_cv_image = cv2.GaussianBlur(open_cv_image, (3, 3), 0)
        final_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))

        return final_image
    except Exception as e:
        print(f"Ön işleme hatası: {e}, orijinal görsel kullanılıyor.")
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def run_inference(image: Image.Image) -> dict:
    current_model = get_model()
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = current_model(input_tensor)
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
    init_db()
    
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL;"))
            conn.commit()
        print("Veritabanı soft delete sütun kontrolü başarıyla tamamlandı.")
    except Exception as e:
        print(f"Sütun kontrolü sırasında bilgi/uyarı: {e}")

@app.get("/")
def root():
    return {"status": "ok", "message": "Hops Disease Detection API çalışıyor"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "running"}

# Endpoint 1: Dosya yükleme (Orijinal + İşlenmiş Görsel URL Desteği)
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Görsel okunamadı")

    # 1. Orijinal dosyayı kaydet
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_save_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_save_path, "wb") as f:
        f.write(contents)

    base_url = "https://plant-disease-api-gicm.onrender.com"
    full_image_url = f"{base_url}/uploads/{unique_filename}"

    # 2. Ön işleme yap (Görseli iyileştir)
    processed_image = enhance_and_preprocess_image(contents)

    # 3. İşlenmiş görseli de sunucuya kaydet
    processed_filename = f"processed_{unique_filename}"
    processed_save_path = os.path.join(UPLOAD_DIR, processed_filename)
    processed_image.save(processed_save_path, quality=95)
    full_processed_image_url = f"{base_url}/uploads/{processed_filename}"

    # 4. Tahmini çalıştır
    try:
        result = run_inference(processed_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

    # 5. Veritabanına kaydet
    record = ScanHistory(
        user_id=current_user.id,
        prediction_class=result["prediction"],
        confidence_score=result["confidence"],
        image_url=full_image_url,
    )
    db.add(record)
    db.commit()

    # 6. Yanıta orijinal ve işlenmiş görsel URL'lerini ekle
    result["image_url"] = full_image_url
    result["processed_image_url"] = full_processed_image_url
    return result

# Endpoint 2: URL'den görsel
class ImageURLRequest(BaseModel):
    image_url: str

@app.post("/predict-url")
async def predict_url(
    body: ImageURLRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        response = requests.get(body.image_url, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel indirilemedi: {str(e)}")

    processed_image = enhance_and_preprocess_image(image_bytes)

    # İşlenmiş görseli URL senaryosu için de kaydedelim
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    processed_filename = f"processed_{unique_filename}"
    processed_save_path = os.path.join(UPLOAD_DIR, processed_filename)
    processed_image.save(processed_save_path, quality=95)

    base_url = "https://plant-disease-api-gicm.onrender.com"
    full_processed_image_url = f"{base_url}/uploads/{processed_filename}"

    try:
        result = run_inference(processed_image)
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

    result["image_url"] = body.image_url
    result["processed_image_url"] = full_processed_image_url
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

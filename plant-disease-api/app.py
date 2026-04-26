from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import urllib.request

app = Flask(__name__)

# Sınıf isimleri
CLASS_NAMES = [
    "Disease-Downy",
    "Disease-Powdery",
    "Healthy",
    "Insect-Pest"
]

# Model yükleme
model = torch.jit.load("efficientnet_plant.pt", map_location="cpu")
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Root test
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Plant Disease API çalışıyor!"})


# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():

    image_data = None

    # 1️⃣ JSON (FlutterFlow image_url buradan gelecek)
    if request.is_json:
        data = request.get_json()

        if "image_url" in data:
            url = data["image_url"]
            image_data = urllib.request.urlopen(url).read()

        elif "image" in data:
            img_str = data["image"]

            if img_str.startswith("http"):
                image_data = urllib.request.urlopen(img_str).read()

            elif "," in img_str:
                image_data = base64.b64decode(img_str.split(",")[1])

            else:
                image_data = base64.b64decode(img_str)

    # 2️⃣ FILE UPLOAD (FlutterFlow veya Postman file gönderirse)
    elif 'image' in request.files:
        image_data = request.files['image'].read()

    elif 'file' in request.files:
        image_data = request.files['file'].read()

    # ❌ hiç veri yoksa
    if image_data is None:
        return jsonify({"error": "Görsel bulunamadı"}), 400

    try:
        # Görseli aç
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Tensor
        tensor = transform(img).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]

        # JSON response
        return jsonify({
            "prediction": pred_class,
            "confidence": float(probs[pred_idx].item()),

            "downyScore": float(probs[0].item()),
            "powderyScore": float(probs[1].item()),
            "healthyScore": float(probs[2].item()),
            "insectScore": float(probs[3].item())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# RENDER PORT FIX
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

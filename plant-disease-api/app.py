from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Sınıflar
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

# Model yükle
model = torch.jit.load("efficientnet_plant.pt", map_location="cpu")
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Ana endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Plant Disease API çalışıyor!'})

# Tahmin endpoint
@app.route('/predict', methods=['POST'])
def predict():

    image_data = None

    # 1. JSON (base64 veya URL)
    if request.is_json and 'image' in request.json:
        image_str = request.json['image']

        if image_str.startswith('http'):
            import urllib.request
            with urllib.request.urlopen(image_str) as response:
                image_data = response.read()

        elif ',' in image_str:
            image_data = base64.b64decode(image_str.split(',')[1])

        else:
            image_data = base64.b64decode(image_str)

    # 2. Multipart (FlutterFlow için en önemli)
    elif 'image' in request.files:
        image_data = request.files['image'].read()

    elif 'file' in request.files:
        image_data = request.files['file'].read()

    if image_data is None:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]

        return jsonify({
            "prediction": pred_class,
            "confidence": float(probs[pred_idx].item()),
            "downyScore": float(probs[0].item()),
            "powderyScore": float(probs[1].item()),
            "healthyScore": float(probs[2].item()),
            "insectScore": float(probs[3].item())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Render için PORT ayarı (ÇOK KRİTİK)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS hatalarını engellemek için
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

app = Flask(__name__)
CORS(app)  # FlutterFlow'dan gelen farklı kökenli istekleri kabul eder

# Sınıf isimleri
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

# Modeli yükleme
try:
    model = torch.jit.load("efficientnet_plant.pt", map_location="cpu")
    model.eval()
    print("✅ Model Başarıyla Yüklendi")
except Exception as e:
    print(f"❌ Model Yükleme Hatası: {e}")

# Görüntü dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'online', 'message': 'Hops API Hazır!'})

@app.route('/predict', methods=['POST', 'GET']) # GET desteği de ekledik test için
def predict():
    image_url = None

    # 1. Adım: Veriyi yakalama (En esnek yöntem)
    if request.method == 'POST':
        # Önce JSON kontrolü
        data = request.get_json(force=True, silent=True)
        if data:
            image_url = data.get('image_url')
        
        # JSON değilse Form verisi kontrolü
        if not image_url:
            image_url = request.form.get('image_url')
            
    # 2. Adım: URL parametresi kontrolü (Yedek plan)
    if not image_url:
        image_url = request.args.get('image_url')

    if not image_url:
        return jsonify({
            'error': 'image_url bulunamadı!',
            'received_data': str(request.data),
            'method': request.method
        }), 400

    try:
        # Resmi indir
        response = requests.get(image_url, timeout=15)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Modeli çalıştır
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = torch.argmax(probs).item()
            
        return jsonify({
            'prediction': CLASS_NAMES[idx],
            'confidence': round(float(probs[idx].item()), 4),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

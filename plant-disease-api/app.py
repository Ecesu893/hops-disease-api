from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Flask objesini en başta tanımlıyoruz (Hata almamak için kritik)
app = Flask(__name__)

CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

# Modeli CPU üzerinde yüklüyoruz (Render Free Tier uyumu için)
try:
    model = torch.jit.load("efficientnet_plant.pt", map_location="cpu")
    model.eval()
    print("✅ Model başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yükleme hatası: {str(e)}")

# Görüntü işleme adımları
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Hops Disease API çalışıyor!'})

@app.route('/predict', methods=['POST'])
def predict():
    # FlutterFlow'dan gelecek JSON verisini kontrol et
    data = request.json
    image_url = data.get('image_url') if data else None

    try:
        # 1. Senaryo: URL gönderildiyse (Supabase üzerinden)
        if image_url:
            resp = requests.get(image_url, timeout=10)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        
        # 2. Senaryo: Direkt dosya gönderildiyse (Multipart form)
        elif 'file' in request.files:
            file = request.files['file']
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        else:
            return jsonify({'error': 'Resim URL veya dosya bulunamadı'}), 400

        # Model tahmini
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
            prediction = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx].item())

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

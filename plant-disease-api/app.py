import torch
import io
import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# --- YAPILANDIRMA ---
device = torch.device('cpu')
MODEL_PATH = 'efficientnet_plant.pt'

CLASS_NAMES = [
    "Disease-Downy",
    "Disease-Powdery",
    "Healthy",
    "Insect-Pest"
]

REMEDIES = {
    "Disease-Downy": "Bakır bazlı fungisitler uygulayın ve yaprak nemini azaltın.",
    "Disease-Powdery": "Kükürt içerikli ilaçlar kullanın ve hava sirkülasyonunu artırın.",
    "Healthy": "Bitki sağlıklı! Düzenli gözleme devam edin.",
    "Insect-Pest": "Zararlı böcek tespit edildi. Biyoteknik tuzaklar veya uygun ilaçlama yapın."
}

# --- MODEL YÜKLEME ---
# Modelin yüklendiğini loglarda görmek için print ekledik
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print("✓ Model başarıyla CPU üzerinde yüklendi.")
except Exception as e:
    print(f"X Model yüklenirken hata oluştu: {e}")

# --- GÖRÜNTÜ ÖN İŞLEME ---
def transform_image(image_bytes):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

# --- ANA SAYFA (404 Hatasını Gidermek İçin) ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>Şerbetçiotu Hastalık Tespit API'si Aktif</h1><p>Tahmin için <b>/predict</b> ucuna POST isteği gönderin.</p>"

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    # FlutterFlow 'file' isminde bir multipart veri göndermeli
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı. Lütfen "file" anahtarı ile bir resim gönderin.'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    try:
        input_tensor = transform_image(img_bytes)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, index = torch.max(probabilities, 0)
        
        result_class = CLASS_NAMES[index.item()]
        
        return jsonify({
            'status': 'success',
            'prediction': result_class,
            'confidence': f"{confidence.item() * 100:.2f}%",
            'remedy': REMEDIES.get(result_class, "Bilgi bulunamadı.")
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- PORT AYARI ---
if __name__ == '__main__':
    # Render PORT çevre değişkenini kullanır, yoksa 5000'de çalışır
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

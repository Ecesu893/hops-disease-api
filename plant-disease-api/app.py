from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

app = Flask(__name__)

# Modeli yükle
# "efficientnet_plant.pt" dosyasının script ile aynı klasörde olduğundan emin ol.
model = torch.jit.load('efficientnet_plant.pt')
model.eval()

# Senin belirlediğin sınıflar
CLASS_NAMES = [
    "Disease-Downy",
    "Disease-Powdery",
    "Healthy",
    "Insect-Pest"
]

# Görüntü hazırlama (EfficientNet standartları)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Resim yüklenmedi'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Görüntüyü modele hazırla
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    # Olasılıkları hesapla
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, index = torch.max(probabilities, 0)
    
    result_class = CLASS_NAMES[index.item()]
    confidence_score = float(confidence.item())

    # FlutterFlow'a dönecek olan JSON verisi
    return jsonify({
        'prediction': result_class,
        'confidence': f"{confidence_score * 100:.2f}%",
        'status': "Success"
    })

if __name__ == '__main__':
    # Localde test için 5000 portu, '0.0.0.0' dış erişime izin verir.
    app.run(host='0.0.0.0', port=5000)

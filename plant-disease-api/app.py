from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import io

app = Flask(__name__)

CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]

model = torch.jit.load("efficientnet_plant.pt", map_location="cpu")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Plant Disease API çalışıyor!'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    file  = request.files['file']
    image = file.read()

    try:
        img    = Image.open(io.BytesIO(image)).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            pred_idx   = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item() * 100

        return jsonify({
            'prediction' : pred_class,
            'confidence' : f"{confidence:.2f}%",
            'all_scores' : {
                CLASS_NAMES[i]: f"{probs[i].item()*100:.2f}%"
                for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
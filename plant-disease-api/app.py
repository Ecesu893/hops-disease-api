from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import urllib.request

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
    print("📥 İstek geldi!")
    print("Files:", request.files)
    print("Form:", request.form)
    print("JSON:", request.json)

    image_data = None

    # JSON body'den al
    if request.json and 'image' in request.json:
        image_str = request.json['image']
        print(f"📝 JSON image uzunluk: {len(image_str) if image_str else 0}")
        
        if image_str and image_str.startswith('http'):
            with urllib.request.urlopen(image_str) as response:
                image_data = response.read()
            print(f"✅ URL'den indirildi: {len(image_data)} bytes")
        elif image_str and ',' in image_str:
            image_data = base64.b64decode(image_str.split(',')[1])
            print(f"✅ Base64 data URL'den alındı: {len(image_data)} bytes")
        elif image_str and len(image_str) > 100:
            image_data = base64.b64decode(image_str)
            print(f"✅ Base64'ten alındı: {len(image_data)} bytes")

    # Multipart form'dan al
    elif 'file' in request.files and request.files['file'].filename != '':
        image_data = request.files['file'].read()
        print(f"✅ Files'tan alındı: {len(image_data)} bytes")

    # Form string'den al
    elif 'file' in request.form and request.form['file'] != '':
        file_val = request.form['file']
        if file_val.startswith('http'):
            with urllib.request.urlopen(file_val) as response:
                image_data = response.read()
        elif ',' in file_val:
            image_data = base64.b64decode(file_val.split(',')[1])
        elif len(file_val) > 100:
            image_data = base64.b64decode(file_val)

    if image_data is None:
        print("❌ Görsel verisi bulunamadı!")
        return jsonify({'error': 'Görsel verisi bulunamadı'}), 400

    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(img).un

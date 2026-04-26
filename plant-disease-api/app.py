from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

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
        print(f"📝 JSON image: {image_str[:100] if image_str else 'boş'}")
        if image_str and image_str.startswith('http'):
            import urllib.request
            with urllib.request.urlopen(image_str) as response:
                image_data = response.read()
            print(f"✅ URL'den indirildi: {len(image_data)} bytes")
        elif image_str and ',' in image_str:
            image_data = base64.b64decode(image_str.split(',')[1])
            print(f"✅ Base64'ten alındı: {len(image_data)} bytes")
        elif image_str:
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
            import urllib.request
            with urllib.request.urlopen(file_val) as response:
                image_data = response.read()
        elif ',' in file_val:
            image_data = base64.b64decode(file_val.split(',')[1])
        else:
            image_data = base64.b64decode(file_val)

    if image_data is None:
        print("❌ Dosya yok!")
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    try:
        img    = Image.open(io.BytesIO(image_data)).convert("RGB")
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
        print(f"❌ Hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

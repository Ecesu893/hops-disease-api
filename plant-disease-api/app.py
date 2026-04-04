from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Model ve sınıf isimleri
CLASS_NAMES = ["Disease-Downy", "Disease-Powdery", "Healthy", "Insect-Pest"]
session     = ort.InferenceSession("efficientnet_plant.onnx")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    img  = img.transpose(2, 0, 1)   # HWC → CHW
    img  = np.expand_dims(img, 0)   # batch boyutu ekle
    return img.astype(np.float32)

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
        input_data = preprocess(image)
        outputs    = session.run(None, {'input': input_data})
        scores     = outputs[0][0]
        pred_idx   = int(np.argmax(scores))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(np.exp(scores[pred_idx]) /
                           np.sum(np.exp(scores)) * 100)

        return jsonify({
            'prediction' : pred_class,
            'confidence' : f"{confidence:.2f}%",
            'all_scores' : {
                CLASS_NAMES[i]: f"{float(np.exp(scores[i]) / np.sum(np.exp(scores)) * 100):.2f}%"
                for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
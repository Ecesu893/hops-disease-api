import requests # Bunu en üste ekle (requirements.txt'ye de ekle)

@app.route('/predict', methods=['POST'])
def predict():
    # JSON içinden URL gelip gelmediğini kontrol et
    data = request.json
    image_url = data.get('image_url') if data else None

    try:
        if image_url:
            # URL'den resmi indir
            resp = requests.get(image_url)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            print(f"✅ URL'den resim indirildi: {image_url}")
        elif 'file' in request.files:
            # Dosya olarak geldiyse
            file = request.files['file']
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            print("✅ Dosya form üzerinden alındı")
        else:
            return jsonify({'error': 'Resim URL veya dosya bulunamadı'}), 400

        # Model İşlemleri (Aynı kalıyor)
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item() # Yüzdeyi FlutterFlow'da da hesaplayabilirsin

        return jsonify({
            'prediction': pred_class,
            'confidence': confidence, # Sayısal değer göndermek veritabanı için daha iyidir
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

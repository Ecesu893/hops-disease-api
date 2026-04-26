from PIL import Image
import io
import base64
import urllib.request

app = Flask(__name__)

@@ -29,26 +28,44 @@ def predict():
print("📥 İstek geldi!")
print("Files:", request.files)
print("Form:", request.form)
    print("JSON:", request.json)

image_data = None

    if 'file' in request.files and request.files['file'].filename != '':
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
        print(f"📝 Form değeri: {file_val[:100]}")
if file_val.startswith('http'):
            import urllib.request
with urllib.request.urlopen(file_val) as response:
image_data = response.read()
            print(f"✅ URL'den indirildi: {len(image_data)} bytes")
elif ',' in file_val:
image_data = base64.b64decode(file_val.split(',')[1])
            print(f"✅ Base64'ten alındı: {len(image_data)} bytes")
else:
image_data = base64.b64decode(file_val)
            print(f"✅ Base64'ten alındı: {len(image_data)} bytes")
    else:

    if image_data is None:
print("❌ Dosya yok!")
return jsonify({'error': 'Dosya bulunamadı'}), 400

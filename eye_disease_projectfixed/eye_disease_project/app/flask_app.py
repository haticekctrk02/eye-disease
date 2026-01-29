import os
import sys
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import cv2
import numpy as np
from torchvision import transforms
from werkzeug.utils import secure_filename

# Ensure project root is on sys.path so local modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grad_cam import GradCAM
from model import create_model

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-with-secure-key'
# Maksimum yükleme boyutu: 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Load model if available
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'best.pth')
CLASSES_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'classes.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
gcam = None
classes = []


def load_classes_from_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    return []


def load_model():
    global model, gcam, classes
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            classes = ckpt.get('classes', None) or load_classes_from_file(CLASSES_PATH)
            num_classes = len(classes) if classes else 10
            model = create_model(num_classes, pretrained=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)
            model.eval()
            gcam = GradCAM(model)
            print(f'Model yüklendi. Sınıf sayısı: {len(classes)}')
        except Exception as e:
            print('Model yükleme hatası:', e)
            model = None
            gcam = None
    else:
        print('Uyarı: models/best.pth bulunamadı. Önce eğitip modeli kaydedin.')

# İlk model yükleme denemesi
load_model()

@app.route('/status')
def status():
    return {
        'model_loaded': model is not None,
        'num_classes': len(classes) if classes else 0
    }

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        load_model()
        return {'ok': True, 'model_loaded': model is not None, 'num_classes': len(classes) if classes else 0}
    except Exception as e:
        return {'ok': False, 'error': str(e)}, 500

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ALLOWED_EXT = {'png','jpg','jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('Dosya bulunamadı')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('Dosya seçilmedi')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, unique)
        file.save(save_path)

        try:
            # Read and preprocess
            img = cv2.imread(save_path)
            if img is None:
                flash('Görüntü okunamadı veya bozuk dosya.')
                return redirect(url_for('index'))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_t = preprocess(img_rgb).unsqueeze(0).to(device)

            if model is None:
                flash('Model bulunamadı. Lütfen önce modeli eğitin ve models/best.pth olarak kaydedin.')
                return redirect(url_for('index'))

            with torch.no_grad():
                out = model(input_t)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idxs = probs.argsort()[::-1]
            display_classes = classes if classes else [str(i) for i in range(len(probs))]
            preds = [(display_classes[i], float(probs[i])) for i in idxs[:5]]

            # Grad-CAM
            cam = gcam(input_t)
            cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)
            out_name = f"out_{uuid.uuid4().hex}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, overlay_bgr)

            # Format probabilities for template
            preds_formatted = [(p[0], f"{p[1]*100:.2f}%") for p in preds]
            return render_template('result.html', preds=preds_formatted, filename=unique, outname=out_name)
        except Exception as e:
            flash(f'Hata: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Desteklenmeyen dosya tipi')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
👁️ Eye Disease Classification System

Deep Learning Based Medical Image Analysis with Grad-CAM

Bu proje, retina (fundus) görüntüleri üzerinden göz hastalıklarını derin öğrenme kullanarak sınıflandırmayı amaçlayan uçtan uca bir sistemdir. Model eğitimi, değerlendirme, görselleştirme (Grad-CAM) ve kullanıcı arayüzü (Flask & PyQt) tek bir projede birleştirilmiştir.

📌 Proje Özeti

🧠 CNN tabanlı derin öğrenme modeli

🔍 Grad-CAM ile açıklanabilir yapay zekâ

📊 Detaylı performans analizi ve raporlama

🌐 Web (Flask) ve Desktop (PyQt) arayüz

📂 Modüler, genişletilebilir proje yapısı

Bu proje özellikle medikal görüntü işleme, sağlıkta yapay zekâ ve açıklanabilir AI (XAI) alanlarına yöneliktir.

🧠 Model & Yaklaşım

Framework: PyTorch

Model tipi: Convolutional Neural Network (CNN)

Eğitim:

train.py

Eğitim sırasında en iyi model models/best.pth olarak kaydedilir

Sınıf etiketleri:

models/classes.txt

📂 Proje Dizini
eye_disease_project/
│
├── train.py                # Model eğitimi
├── eval.py                 # Model değerlendirme
├── model.py                # CNN mimarisi
├── check_model.py          # Model test / sanity check
├── grad_cam.py             # Grad-CAM görselleştirme
├── requirements.txt        # Bağımlılıklar
│
├── models/
│   ├── best.pth            # Eğitilmiş model
│   └── classes.txt         # Sınıf isimleri
│
├── data/                   # Veri seti (harici / eklenebilir)
│
├── reports/
│   ├── classification_report.json
│   ├── confusion_matrix.png
│   └── project_report_draft.md
│
├── notebooks/
│   └── quick_start.ipynb   # Hızlı başlangıç notebook’u
│
└── app/
    ├── flask_app.py        # Web arayüz (Flask)
    ├── pyqt_app.py         # Desktop arayüz (PyQt)
    ├── templates/
    │   ├── index.html
    │   └── result.html
    └── static/
        ├── uploads/
        └── outputs/

⚙️ Kurulum
1️⃣ Ortam Oluşturma
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Bağımlılıklar
pip install -r requirements.txt

🚀 Kullanım
🔹 Model Eğitimi
python train.py

🔹 Model Değerlendirme
python eval.py

🔹 Grad-CAM Görselleştirme
python grad_cam.py --image path/to/image.jpg

🌐 Web Arayüz (Flask)
python app/flask_app.py


Tarayıcı:

http://127.0.0.1:5000


Özellikler

Görüntü yükleme

Tahmin sonucu

Grad-CAM ısı haritası

🖥️ Desktop Arayüz (PyQt)
python app/pyqt_app.py


Özellikler

Masaüstü uygulaması

Görsel seçimi

Anlık tahmin ve görselleştirme

📊 Performans & Raporlama

✔️ Confusion Matrix

✔️ Precision / Recall / F1-Score

✔️ JSON formatında detaylı rapor

✔️ Grad-CAM ile model karar açıklaması

Çıktılar:

reports/
 ├── classification_report.json
 └── confusion_matrix.png

🔍 Açıklanabilir Yapay Zekâ (Grad-CAM)

Grad-CAM sayesinde modelin:

Görüntünün hangi bölgelerine odaklandığı

Kararın hangi görsel ipuçlarına dayandığı

net biçimde analiz edilebilir.

Bu özellik özellikle medikal güvenilirlik açısından kritiktir.

🎯 Kullanım Alanları

Sağlıkta karar destek sistemleri

Medikal görüntü analizi

Akademik projeler & bitirme tezleri

Yapay zekâ + sağlık uygulamaları

⚠️ Uyarı

Bu proje akademik ve araştırma amaçlıdır.
Tıbbi teşhis yerine geçmez.

👩‍💻 Geliştirici

Hatice Kocatürk

Deep Learning • Computer Vision • Medical AI

⭐ Geliştirme Önerileri

YOLO tabanlı lezyon tespiti

Multi-label classification

Model ensemble

Docker desteği

REST API

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

eye_disease_project/
│
├── train.py
│   └── Modelin eğitildiği ana dosya
│
├── eval.py
│   └── Eğitilmiş modelin test ve değerlendirme işlemleri
│
├── model.py
│   └── CNN mimarisinin tanımlandığı dosya
│
├── check_model.py
│   └── Modelin hızlı test / sanity check işlemleri
│
├── grad_cam.py
│   └── Grad-CAM tabanlı açıklanabilirlik (XAI) görselleştirmeleri
│
├── requirements.txt
│   └── Projede kullanılan Python bağımlılıkları
│
├── models/
│   ├── best.pth
│   │   └── Eğitilmiş en iyi model ağırlıkları
│   │
│   └── classes.txt
│       └── Modelin sınıflandırdığı hastalık sınıfları
│
├── data/
│   └── Veri seti dizini (harici olarak eklenebilir)
│
├── reports/
│   ├── classification_report.json
│   │   └── Precision, Recall, F1-score gibi metrikler
│   │
│   ├── confusion_matrix.png
│   │   └── Model performansını gösteren Confusion Matrix
│   │
│   └── project_report_draft.md
│       └── Proje raporu taslağı
│
├── notebooks/
│   └── quick_start.ipynb
│       └── Hızlı deneme ve analizler için Jupyter Notebook
│
└── app/
    ├── flask_app.py
    │   └── Web tabanlı kullanıcı arayüzü (Flask)
    │
    ├── pyqt_app.py
    │   └── Masaüstü uygulaması (PyQt)
    │
    ├── templates/
    │   ├── index.html
    │   │   └── Ana sayfa
    │   └── result.html
    │       └── Tahmin sonuçları sayfası
    │
    └── static/
        ├── uploads/
        │   └── Kullanıcı tarafından yüklenen görüntüler
        │
        └── outputs/
            └── Grad-CAM ve model çıktı görselleri


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

import sys
import os
import torch
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QListWidget, QProgressBar, QMessageBox, QMenuBar, QAction, QStatusBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms

# Ensure project root is on sys.path so local modules are importable when running from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import create_model
from grad_cam import GradCAM


class MainWindow(QWidget):
    def __init__(self, model_path=None, classes=None):
        super().__init__()
        self.setWindowTitle('Göz Hastalıkları - Karar Destek Sistemi')
        self.resize(900, 600)
        layout = QHBoxLayout()

        left = QVBoxLayout()
        self.img_label = QLabel('Görüntü yok')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(512,512)
        left.addWidget(self.img_label)

        # Enable drag & drop
        self.setAcceptDrops(True)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton('Görüntü Yükle')
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)

        self.load_model_btn = QPushButton('Model Yükle')
        self.load_model_btn.clicked.connect(self.load_model_from_dialog)
        btn_layout.addWidget(self.load_model_btn)

        self.save_cam_btn = QPushButton('Grad-CAM Kaydet')
        self.save_cam_btn.clicked.connect(self.save_cam)
        self.save_cam_btn.setEnabled(False)
        btn_layout.addWidget(self.save_cam_btn)

        left.addLayout(btn_layout)

        right = QVBoxLayout()
        self.result_list = QListWidget()
        right.addWidget(QLabel('Tahminler (olasılık)'))
        right.addWidget(self.result_list)

        self.cam_label = QLabel('Grad-CAM')
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setFixedSize(256,256)
        right.addWidget(self.cam_label)

        layout.addLayout(left)
        layout.addLayout(right)

        # Status bar
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Model yüklenmedi' if self.model is None else f'Model yüklendi ({len(self.classes)} sınıf)')
        main_layout.addWidget(self.status_bar)
        self.setLayout(main_layout)

        # Model yükle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            classes_ckpt = ckpt.get('classes', classes)
            self.classes = classes_ckpt
            num_classes = len(self.classes)
            self.model = create_model(num_classes, pretrained=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            # GradCAM
            self.gcam = GradCAM(self.model)
        else:
            self.model = None
            self.classes = classes

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Görüntü seç', os.getcwd(), 'Images (*.png *.jpg *.jpeg)')
        if not fname:
            return
        img = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio)
        self.img_label.setPixmap(pixmap)

        if self.model:
            self.predict_and_show(img_rgb)

    def predict_and_show(self, img_rgb):
        input_t = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        # Show top-3
        self.result_list.clear()
        idxs = np.argsort(probs)[::-1]
        for i in idxs[:5]:
            self.result_list.addItem(f"{self.classes[i]}: {probs[i]:.3f}")

        # Grad-CAM
        cam = self.gcam(input_t)
        cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        # Store overlay & enable save button
        self.last_overlay = overlay_rgb
        self.save_cam_btn.setEnabled(True)
        h,w,ch = overlay_rgb.shape
        qimg2 = QImage(overlay_rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix2 = QPixmap.fromImage(qimg2).scaled(self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio)
        self.cam_label.setPixmap(pix2)
        # Enable saving of the last overlay
        self.last_overlay = None
        self.save_cam_btn.setEnabled(False)

    def load_model_from_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Model dosyası seç', os.path.join(os.getcwd(), '..', 'models'), 'PyTorch Model (*.pth)')
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            classes_ckpt = ckpt.get('classes', None)
            if not classes_ckpt and os.path.exists(os.path.join(os.path.dirname(path), 'classes.txt')):
                with open(os.path.join(os.path.dirname(path), 'classes.txt'), 'r', encoding='utf-8') as f:
                    classes_ckpt = [l.strip() for l in f if l.strip()]
            if not classes_ckpt:
                QMessageBox.warning(self, 'Uyarı', 'Sınıf listesi bulunamadı; sınıflar sayısal etiketlerle gösterilecek.')
                classes_ckpt = []
            self.classes = classes_ckpt
            num_classes = len(self.classes) if self.classes else 10
            self.model = create_model(num_classes, pretrained=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.gcam = GradCAM(self.model)
            self.status_bar.showMessage(f'Model yüklendi ({len(self.classes)} sınıf)')
            QMessageBox.information(self, 'Model', 'Model başarıyla yüklendi.')
        except Exception as e:
            QMessageBox.critical(self, 'Hata', f'Model yüklenirken hata: {e}')

    def save_cam(self):
        if self.last_overlay is None:
            QMessageBox.warning(self, 'Uyarı', 'Kaydedilecek Grad-CAM yok.')
            return
        fname, _ = QFileDialog.getSaveFileName(self, 'Grad-CAM kaydet', os.getcwd(), 'PNG (*.png)')
        if not fname:
            return
        # last_overlay is RGB uint8
        cv2.imwrite(fname, cv2.cvtColor(self.last_overlay, cv2.COLOR_RGB2BGR))
        QMessageBox.information(self, 'Kaydet', 'Grad-CAM başarıyla kaydedildi.')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            # load image directly
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, 'Uyarı', 'Geçersiz görüntü dosyası.')
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio)
            self.img_label.setPixmap(pixmap)
            if self.model:
                self.predict_and_show(img_rgb)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/best.pth')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(model_path=args.model_path)
    window.show()
    sys.exit(app.exec_())
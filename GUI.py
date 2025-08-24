import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from predict import predict_image
import cv2


class PredictionThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, image_path, yolo_model, classifier, idx_to_label, device='cpu'):
        super().__init__()
        self.image_path = image_path
        self.yolo = yolo_model
        self.classifier = classifier
        self.idx_to_label = idx_to_label
        self.device = device
    def run(self):
        res = predict_image(self.image_path, self.yolo, self.classifier, self.idx_to_label, device=self.device)
        if res.get('message'):
            self.error.emit(res['message'])
        else:
            self.finished.emit(res)


class CarCropGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Crop & Predict GUI")
        self.setGeometry(100, 100, 900, 500)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = None
        self.classifier = None
        self.idx_to_label = None
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#222; border:1px solid #444; color:#ddd;")
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.result_label.setFont(font)
        self.load_button = QPushButton("Wybierz obraz")
        self.load_button.clicked.connect(self.load_image)
        self.heatmap_button = QPushButton("Pokaż heatmapę")
        self.heatmap_button.clicked.connect(self.show_heatmap)
        self.heatmap_button.setEnabled(False)
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.load_button)
        main_layout.addWidget(self.heatmap_button)
        main_layout.addWidget(self.heatmap_label)
        self.setLayout(main_layout)
        self.current_image = None
        self.current_image_path = None
        self.pred_thread = None
        self.last_result = None
        self.heatmap_data = None
        self.load_models()
    def load_models(self):
        if self.yolo is None:
            from ultralytics import YOLO
            y = YOLO('yolov8s.pt')
            try:
                y.to('cpu')
            except Exception:
                pass
            self.yolo = y
        # classifier i idx_to_label będą ładowane automatycznie przez predict_image
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Wybierz obraz', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd', f'Nie można otworzyć obrazu: {e}')
            return
        self.current_image = img
        self.current_image_path = path
        pix = QPixmap(path)
        self.image_label.setPixmap(pix.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.result_label.setText('')
        if self.yolo is None:
            self.load_models()
        self.result_label.setText('Predicting...')
        # classifier i idx_to_label przekazujemy jako None, predict_image sam je załaduje
        self.pred_thread = PredictionThread(self.current_image_path, self.yolo, None, None, device=self.device)
        self.pred_thread.finished.connect(self._on_pred_finished)
        self.pred_thread.error.connect(self._on_pred_error)
        self.pred_thread.start()
    def _on_pred_finished(self, res):
        brand = res.get('brand')
        conf = res.get('confidence')
        self.heatmap_data = res.get('heatmap')
        if self.heatmap_data:
            self.heatmap_button.setEnabled(True)
        else:
            self.heatmap_button.setEnabled(False)
            self.heatmap_label.clear()
        if brand is None:
            self.result_label.setText(res.get('message', 'Brak wyników'))
        else:
            self.result_label.setText(f"Brand={brand}, Confidence={conf:.2f}%")

    def show_heatmap(self):
        if not self.heatmap_data:
            self.heatmap_label.clear()
            return
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import QByteArray
        import base64
        img_bytes = base64.b64decode(self.heatmap_data)
        qimg = QPixmap()
        qimg.loadFromData(QByteArray(img_bytes), 'PNG')
        self.heatmap_label.setPixmap(qimg.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def _on_pred_error(self, err):
        self.result_label.setText(f"Prediction error: {err}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CarCropGUI()
    win.show()
    sys.exit(app.exec_())

import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QMessageBox,
    QFrame, QSizePolicy, QGraphicsOpacityEffect
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from predict import predict_image
from database import init_db, insert_record, save_image_copy
from history_viewer import HistoryViewer
import cv2
import time


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
        start = time.time()
        res = predict_image(self.image_path, self.yolo, self.classifier, self.idx_to_label, device=self.device)
        res['processing_time'] = time.time() - start
        if res.get('message'):
            self.error.emit(res['message'])
        else:
            self.finished.emit(res)


class CarCropGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Crop & Predict GUI")
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(900, 640)
        self.setStyleSheet("""
            QWidget { background: #f4f6f8; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 14px; color: #212121; }
            QLabel#TitleLabel { font-size: 22px; font-weight: 700; color: #111; margin-bottom: 8px; }
            QLabel#ResultLabel { font-size: 16px; font-weight: 600; color: #0b72bf; margin: 8px 0 14px 0; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #38e4ae); color: white; border: none; border-radius: 12px; padding: 8px 18px; font-size: 14px; font-weight: 600; margin: 6px; min-width: 140px; min-height: 36px; }
            QPushButton:disabled { background: #bfc7d1; color: #eee; }
            QFrame#ImageFrame { background: #ffffff; border-radius: 12px; border: 1px solid #e6e9ee; padding: 8px; }
            QLabel#HeatmapLabel { border-radius: 10px; }
        """)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = None
        self.classifier = None
        self.idx_to_label = None

        # header
        self.title_label = QLabel("Car Crop & Predict GUI")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # image frame
        self.image_frame = QFrame()
        self.image_frame.setObjectName("ImageFrame")
        self.image_frame.setFixedSize(800, 460)
        self.image_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.image_label = QLabel(self.image_frame)
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(784, 444)
        self.image_label.move(8, 8)

        self.heatmap_label = QLabel(self.image_frame)
        self.heatmap_label.setObjectName("HeatmapLabel")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setFixedSize(self.image_label.size())
        self.heatmap_label.move(8, 8)
        self.heatmap_label.hide()

        # result label
        self.result_label = QLabel("")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)

        # buttons
        self.load_button = QPushButton("Wybierz obraz")
        self.load_button.clicked.connect(self.load_image)
        self.heatmap_button = QPushButton("Pokaż heatmapę")
        self.heatmap_button.setEnabled(False)
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        self.confirm_button = QPushButton("Zgłoś nieprawidłową predykcję")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setVisible(False)
        self.confirm_button.clicked.connect(self._on_confirm_click)
        self.history_button = QPushButton("Historia")
        self.history_button.clicked.connect(self._open_history)

        # layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.image_frame, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(12)
        btns_layout.addStretch(1)
        btns_layout.addWidget(self.load_button)
        btns_layout.addWidget(self.heatmap_button)
        btns_layout.addWidget(self.confirm_button)
        btns_layout.addWidget(self.history_button)
        btns_layout.addStretch(1)
        main_layout.addLayout(btns_layout)

        self.setLayout(main_layout)

        # internal state
        self.current_image = None
        self.current_image_path = None
        self.pred_thread = None
        self.last_result = None
        self.heatmap_data = None
        self.heatmap_visible = False
        self.feedback_saved = False
        self.load_models()
        self.confirm_button.setVisible(False)
        self.confirm_button.setEnabled(False)
        # container for running animations so they are not garbage-collected
        self._animations = []
        # ensure db exists
        try:
            init_db()
        except Exception:
            pass

    def _fade_in_widget(self, widget, duration=300):
        """Apply a fade-in animation to a widget (keeps reference to animation)."""
        try:
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(duration)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.start()
            # keep reference until finished
            self._animations.append(anim)

            def _on_finished():
                try:
                    self._animations.remove(anim)
                except ValueError:
                    pass

            anim.finished.connect(_on_finished)
        except Exception:
            pass

    def load_models(self):
        if self.yolo is None:
            try:
                from ultralytics import YOLO
                y = YOLO('yolov8s.pt')
                try:
                    y.to('cpu')
                except Exception:
                    pass
                self.yolo = y
            except Exception:
                self.yolo = None

    def load_image(self):
        # hide report button while predicting
        try:
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
            self.feedback_saved = False
        except Exception:
            pass

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
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        # animate smooth fade-in for the newly loaded image
        try:
            self._fade_in_widget(self.image_label, duration=400)
        except Exception:
            pass
        # reset/close any existing heatmap and disable the heatmap button until a new one is generated
        try:
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.heatmap_visible = False
            self.heatmap_button.setEnabled(False)
            self.heatmap_button.setText("Pokaż heatmapę")
        except Exception:
            pass

        self.result_label.setText('Predicting...')

        # start prediction thread
        self.pred_thread = PredictionThread(self.current_image_path, self.yolo, None, None, device=self.device)
        self.pred_thread.finished.connect(self._on_pred_finished)
        self.pred_thread.error.connect(self._on_pred_error)
        self.pred_thread.start()

    def _on_pred_finished(self, res):
        brand = res.get('brand')
        conf = res.get('confidence')
        proc_time = res.get('processing_time', 0.0)
        self.heatmap_data = res.get('heatmap')
        if self.heatmap_data:
            self.heatmap_button.setEnabled(True)
        else:
            self.heatmap_button.setEnabled(False)
            self.heatmap_label.clear()
        # store last result
        self.last_result = {'brand': brand, 'confidence': conf, 'image_path': self.current_image_path}
        # auto-save as correct by default (single entry)
        if not self.feedback_saved:
            try:
                entry = {'timestamp': int(time.time()), 'image': self.last_result.get('image_path'), 'predicted': self.last_result.get('brand'), 'confidence': float(self.last_result.get('confidence') or 0.0), 'correct': True}
                hist_path = os.path.join(os.path.dirname(__file__), 'history.jsonl')
                with open(hist_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                self.feedback_saved = True
            except Exception:
                pass
        # show report button
        self.confirm_button.setVisible(True)
        self.confirm_button.setEnabled(True)
        # save a copy of the image and insert into DB
        try:
            # save a copy into the hidden database image folder
            try:
                dst = save_image_copy(self.current_image_path)
            except Exception:
                # fallback: try to save via PIL into the hidden dir
                base = os.path.basename(self.current_image_path)
                dst = save_image_copy(self.current_image_path, prefix='copied')
            insert_record(self.current_image_path, dst, brand, float(conf or 0.0), True, float(proc_time), int(time.time()))
        except Exception:
            pass
        if brand is None:
            self.result_label.setText(res.get('message', 'Brak wyników'))
        else:
            self.result_label.setText(f"Brand={brand}, Confidence={conf:.2f}%")

    def toggle_heatmap(self):
        if not self.heatmap_data:
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            return
        if self.heatmap_visible:
            self.heatmap_label.hide()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            return
        import base64
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import QByteArray
        try:
            img_bytes = base64.b64decode(self.heatmap_data)
            qimg = QPixmap()
            qimg.loadFromData(QByteArray(img_bytes), 'PNG')
            target = self.image_label.size()
            scaled = qimg.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(target)
            canvas.fill(Qt.transparent)
            from PyQt5.QtGui import QPainter
            painter = QPainter(canvas)
            x = (canvas.width() - scaled.width()) // 2
            y = (canvas.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
            self.heatmap_label.setPixmap(canvas)
            # set initial invisible and animate fade-in
            self.heatmap_label.setVisible(True)
            try:
                self._fade_in_widget(self.heatmap_label, duration=350)
            except Exception:
                pass
            self.heatmap_button.setText("Schowaj heatmapę")
            self.heatmap_visible = True
        except Exception as e:
            print(f"show heatmap error: {e}")

    def show_heatmap(self):
        self.toggle_heatmap()

    def _on_pred_error(self, err):
        self.result_label.setText(f"Prediction error: {err}")
        try:
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
        except Exception:
            pass

    def _open_history(self):
        try:
            hv = HistoryViewer(self)
            hv.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można otworzyć historii: {e}')

    def _on_confirm_click(self):
        # user reports incorrect prediction
        if not getattr(self, 'last_result', None):
            QMessageBox.information(self, 'Info', 'Brak wyniku do zgłoszenia')
            return
        try:
            entry = {'timestamp': int(time.time()), 'image': self.last_result.get('image_path'), 'predicted': self.last_result.get('brand'), 'confidence': float(self.last_result.get('confidence') or 0.0), 'correct': False, 'reported': True}
            hist_path = os.path.join(os.path.dirname(__file__), 'history.jsonl')
            with open(hist_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            QMessageBox.information(self, 'Dziękuję', 'Zgłoszenie zapisane')
            self.confirm_button.setEnabled(False)
            self.confirm_button.setVisible(False)
            self.feedback_saved = True
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można zapisać zgłoszenia: {e}')

    def closeEvent(self, event):
        # wait for prediction thread to finish or terminate it to avoid crashes
        if getattr(self, 'pred_thread', None) is not None and self.pred_thread.isRunning():
            reply = QMessageBox.question(self, 'Zamykanie', 'Predykcja w toku. Poczekać na zakończenie?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.pred_thread.wait(10000)
            else:
                try:
                    self.pred_thread.terminate()
                except Exception:
                    pass
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CarCropGUI()
    win.show()
    sys.exit(app.exec_())

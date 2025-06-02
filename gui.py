import sys
import subprocess
import time
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
from database import Database
import os

class CarRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Recognition GUI")
        self.setFixedSize(420, 600)
        self.setStyleSheet("background-color: #f7f7f7;")

        # Inicjalizacja bazy danych
        self.db = Database()

        # Główny kontener z marginesami
        main_container = QWidget()
        main_container.setContentsMargins(20, 20, 20, 20)
        
        # Nagłówek
        self.header = QLabel("Rozpoznawanie marki samochodu")
        self.header.setFont(QFont("Arial", 18, QFont.Bold))
        self.header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.header.setStyleSheet("margin-bottom: 10px;")

        # Opis
        self.desc = QLabel("Wybierz zdjęcie auta, aby rozpoznać markę.")
        self.desc.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.desc.setStyleSheet("color: #555; margin-bottom: 20px;")

        # Podgląd zdjęcia w ramce
        self.img_frame = QFrame()
        self.img_frame.setFrameShape(QFrame.Box)
        self.img_frame.setLineWidth(2)
        self.img_frame.setStyleSheet("background: #fff; border-radius: 8px;")
        
        # Kontener na zdjęcie
        img_container = QWidget()
        img_container.setFixedSize(340, 220)
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(10, 10, 10, 10)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setFixedSize(320, 200)
        img_layout.addWidget(self.img_label)
        self.img_frame.setLayout(img_layout)
        
        # Przycisk z ikoną
        self.button = QPushButton(" Wybierz plik")
        self.button.setIcon(QIcon.fromTheme("document-open"))
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; font-size: 16px;
                border-radius: 6px; padding: 10px 24px; margin: 20px 0;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)
        self.button.clicked.connect(self.choose_file)
        
        # Wynik w kolorowym polu
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            background: #e3f2fd; color: #222; font-size: 16px;
            border-radius: 8px; padding: 16px; margin-top: 20px;
        """)

        # Przycisk historii
        self.history_button = QPushButton("Historia predykcji")
        self.history_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; font-size: 14px;
                border-radius: 6px; padding: 8px 16px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.history_button.clicked.connect(self.show_history)

        # Layout główny
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.header)
        vlayout.addWidget(self.desc)
        vlayout.addWidget(self.img_frame, alignment=Qt.AlignmentFlag.AlignHCenter)
        vlayout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignHCenter)
        vlayout.addWidget(self.result_label)
        vlayout.addWidget(self.history_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        vlayout.addStretch()
        main_container.setLayout(vlayout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_container)
        self.setLayout(main_layout)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.img_label.setPixmap(QPixmap(file_path).scaled(
                320, 200,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation
            ))
            self.result_label.setText("Rozpoznawanie...")
            QApplication.processEvents()
            with open(file_path, "rb") as f:
                files = {"file": f}
                try:
                    r = requests.post("http://localhost:8000/predict", files=files)
                    r.raise_for_status()
                    data = r.json()
                    
                    # Zapisz wynik do bazy danych
                    self.db.save_prediction(
                        image_path=file_path,
                        brand=data['brand'],
                        confidence=data['confidence']
                    )
                    
                    self.result_label.setStyleSheet("""
                        background: #e8f5e9; color: #222; font-size: 18px;
                        border-radius: 8px; padding: 16px; margin-top: 20px;
                    """)
                    self.result_label.setText(
                        f"<b>Marka:</b> {data['brand']}<br>"
                        f"<b>Pewność:</b> {data['confidence']:.2f}%<br>"
                        f"<small>Zapisano w bazie danych</small>"
                    )
                except Exception as e:
                    self.result_label.setStyleSheet("""
                        background: #ffebee; color: #b71c1c; font-size: 16px;
                        border-radius: 8px; padding: 16px; margin-top: 20px;
                    """)
                    self.result_label.setText("Błąd połączenia z serwerem lub predykcji.")

    def show_history(self):
        """Wyświetla okno z historią predykcji"""
        predictions = self.db.get_all_predictions()
        history_text = "Historia predykcji:\n\n"
        for pred in predictions:
            _, img_path, brand, conf, timestamp = pred
            history_text += f"Data: {timestamp}\n"
            history_text += f"Plik: {os.path.basename(img_path)}\n"
            history_text += f"Marka: {brand}\n"
            history_text += f"Pewność: {conf:.2f}%\n"
            history_text += "-" * 40 + "\n"
        
        msg = QMessageBox()
        msg.setWindowTitle("Historia predykcji")
        msg.setText(history_text)
        msg.setStyleSheet("QMessageBox { min-width: 400px; }")
        msg.exec_()

if __name__ == "__main__":
    # Start backend
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app"])
    time.sleep(2)  # Daj serwerowi czas na start
    app = QApplication(sys.argv)
    window = CarRecognitionApp()
    window.show()
    try:
        app.exec_()
    finally:
        server.terminate()
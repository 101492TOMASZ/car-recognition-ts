import sys
import os
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QFileDialog, QFrame, QMessageBox, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
from predict import predict_image
from database import Database
from history_viewer import HistoryViewer

class CarRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.db = Database()  # Initialize database
        self.current_prediction_id = None  # Add this line to store the current prediction ID
        self.init_ui()
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QFrame#preview_frame {
                background-color: #252526;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0px 30px 0px;
            }
        """)

    def init_ui(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Car Recognition")
        self.setFixedSize(800, 800)  # Change from setMinimumSize to setFixedSize

        # Custom top bar with - and X buttons
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        minimize_btn = QPushButton("–")
        minimize_btn.setFixedSize(36, 36)
        minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #252526;
                color: #bdbdbd;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
                color: #ffd600;
            }
        """)
        minimize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        minimize_btn.clicked.connect(self.showMinimized)
        top_bar.addWidget(minimize_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(36, 36)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #252526;
                color: #e57373;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
                color: #ff1744;
            }
        """)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close_window)
        top_bar.addWidget(close_btn)

        # Header
        self.header = QLabel("Rozpoznawanie marki samochodu")
        self.header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Description
        self.desc = QLabel("Wybierz zdjęcie samochodu, aby rozpoznać jego markę")
        self.desc.setFont(QFont("Segoe UI", 12))
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc.setStyleSheet("color: #cccccc; margin: 10px 0px 20px 0px;")

        # Image preview frame
        self.img_frame = QFrame()
        self.img_frame.setObjectName("preview_frame")
        self.img_frame.setContentsMargins(0, 0, 0, 20)  # Add bottom margin
        img_layout = QVBoxLayout(self.img_frame)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_layout.addWidget(self.img_label)
        # Set the label to expand within the frame, but do not set a fixed size
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.img_frame.setFixedSize(720, 300)  # Set fixed size for the frame

        # Buttons layout (horizontal)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 20, 0, 20)  # Add vertical margins
        
        self.button = QPushButton("Wybierz zdjęcie")
        self.button.setFixedWidth(200)  # Set fixed width for button
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.choose_file)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        self.history_button = QPushButton("Historia predykcji")
        self.history_button.setFixedWidth(200)  # Set fixed width for button
        self.history_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.history_button.clicked.connect(self.show_history)
        self.history_button.setStyleSheet("""
            QPushButton {
                background-color: #3b3b3b;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4e4e4e;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
        """)
        
        btn_layout.addStretch()  # Add spacing before buttons
        btn_layout.addWidget(self.button)
        btn_layout.addSpacing(20)  # Add spacing between buttons
        btn_layout.addWidget(self.history_button)
        btn_layout.addStretch()  # Add spacing after buttons

        # Result label with improved contrast
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            background-color: #181818;
            color: #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        """)

        # Feedback buttons
        self.feedback_layout = QHBoxLayout()
        self.feedback_layout.setContentsMargins(0, 10, 0, 0)
        
        self.good_pred_btn = QPushButton("✓ Dobra predykcja")
        self.good_pred_btn.setFixedWidth(200)
        self.good_pred_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.good_pred_btn.clicked.connect(lambda: self.mark_prediction(False))
        self.good_pred_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        
        self.bad_pred_btn = QPushButton("✕ Zła predykcja")
        self.bad_pred_btn.setFixedWidth(200)
        self.bad_pred_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bad_pred_btn.clicked.connect(lambda: self.mark_prediction(True))
        self.bad_pred_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        self.feedback_layout.addStretch()
        self.feedback_layout.addWidget(self.good_pred_btn)
        self.feedback_layout.addSpacing(20)
        self.feedback_layout.addWidget(self.bad_pred_btn)
        self.feedback_layout.addStretch()

        # Hide feedback buttons initially
        self.good_pred_btn.hide()
        self.bad_pred_btn.hide()

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.addLayout(top_bar)
        layout.addWidget(self.header)
        layout.addWidget(self.desc)
        layout.addWidget(self.img_frame)
        layout.addLayout(btn_layout)
        layout.addWidget(self.result_label)
        layout.addLayout(self.feedback_layout)  # Add feedback buttons
        self.setLayout(layout)

        # Variables for window dragging
        self._drag_active = False
        self._drag_position = None

    def close_window(self):
        self.close()

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            # Update image preview
            pixmap = QPixmap(file_path)
            frame_size = self.img_frame.size()
            scaled_pixmap = pixmap.scaled(
                frame_size.width() - 40,  # Account for frame padding
                frame_size.height() - 40,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)

            try:
                # Use predict.py for prediction
                image = Image.open(file_path).convert('RGB')
                result = predict_image(image)
                
                # Save to database and get prediction ID
                self.current_prediction_id = self.db.save_prediction(
                    image_path=file_path,
                    brand=result['brand'],
                    confidence=result['confidence']
                )

                # Update result label
                self.result_label.setStyleSheet("""
                    background-color: #e8f5e9;
                    color: #222;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                """)
                self.result_label.setText(
                    f"Brand: {result['brand']}\n"
                    f"Confidence: {result['confidence']:.2f}%"
                )
                
                # Show feedback buttons after prediction
                self.good_pred_btn.show()
                self.bad_pred_btn.show()

            except ValueError as e:  # Car validation error
                self.current_prediction_id = None
                self.good_pred_btn.hide()
                self.bad_pred_btn.hide()
                self.result_label.setStyleSheet("""
                    background-color: #ffebee;
                    color: #b71c1c;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                """)
                self.result_label.setText(str(e))
            except Exception as e:
                self.current_prediction_id = None
                self.good_pred_btn.hide()
                self.bad_pred_btn.hide()
                self.result_label.setStyleSheet("""
                    background-color: #ffebee;
                    color: #b71c1c;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                """)
                self.result_label.setText(f"Error: {str(e)}")

    def show_prediction_result(self, brand, confidence):
        self.result_label.setStyleSheet("""
            background-color: #0f3622;
            color: #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        """)
        self.result_label.setText(
            f'<div style="font-size: 18px; margin-bottom: 8px;">🚗 <b>{brand}</b></div>'
            f'<div style="color: #cccccc;">Pewność: {confidence:.2f}%</div>'
        )

    def show_error(self, message):
        self.result_label.setStyleSheet("""
            background-color: #442726;
            color: #fff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        """)
        self.result_label.setText(f"❌ {message}")

    def show_history(self):
        """Wyświetla okno z historią predykcji w formie tabeli"""
        predictions = self.db.get_all_predictions()
        viewer = HistoryViewer(predictions, self.db)
        viewer.exec_()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.pos().y() < 60:
            self._drag_active = True
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPos() - self._drag_position)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_active = False
        super().mouseReleaseEvent(event)

    def mark_prediction(self, is_bad):
        if self.current_prediction_id is not None:
            self.db.mark_prediction_as_bad(self.current_prediction_id, is_bad)
            # Hide buttons after feedback
            self.good_pred_btn.hide()
            self.bad_pred_btn.hide()
            # Show confirmation
            if is_bad:
                self.result_label.setStyleSheet("""
                    background-color: #ffebee;
                    color: #c62828;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                """)
                self.result_label.setText("Oznaczono jako złą predykcję")
            else:
                self.result_label.setStyleSheet("""
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 18px;
                    font-weight: bold;
                """)
                self.result_label.setText("Oznaczono jako dobrą predykcję")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CarRecognitionApp()
    ex.show()
    sys.exit(app.exec_())

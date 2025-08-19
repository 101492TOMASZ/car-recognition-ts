import sys
import os
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QFileDialog, QFrame, QMessageBox, QHBoxLayout, QSizePolicy,
    QTableWidget, QTableWidgetItem, QDialog, QHeaderView  # dodaj importy
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QImage  # dodaj import
from PyQt5.QtCore import Qt
from predict import predict_image
from database import Database
from history_viewer import HistoryViewer

class BatchResultDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wyniki batch testu")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Plik", "Marka", "Pewność (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(len(results))
        for i, (filename, result) in enumerate(results.items()):
            self.table.setItem(i, 0, QTableWidgetItem(filename))
            self.table.setItem(i, 1, QTableWidgetItem(str(result.get('brand', 'Błąd'))))
            self.table.setItem(i, 2, QTableWidgetItem(
                f"{result.get('confidence', 0):.2f}" if 'confidence' in result else "-"
            ))
        layout.addWidget(self.table)
        self.setLayout(layout)

class CarRecognitionApp(QWidget):
    def batch_test(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz zdjęcia do batch testu", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_paths:
            return
        self.process_batch_images(file_paths)
    def __init__(self):
        super().__init__()
        self.db = Database()  # Initialize database
        self.current_prediction_id = None  # Add this line to store the current prediction ID
        self.original_image = None  # Przechowuj oryginalny obraz
        self.heatmap_img = None     # Przechowuj heatmapę PIL.Image
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
        self.setMinimumSize(800, 800)  # Minimalny rozmiar
        self.setMaximumSize(1920, 1200)  # Opcjonalnie: ogranicz maksymalny rozmiar

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

        # Dodaj poziomy layout na dwa obrazki: oryginał i heatmapa
        img_layout = QHBoxLayout(self.img_frame)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.heatmap_label = QLabel()  # Nowy label na heatmapę
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.heatmap_label.hide()  # Ukryj heatmapę na start

        img_layout.addWidget(self.img_label)
        img_layout.addSpacing(20)
        img_layout.addWidget(self.heatmap_label)

        self.img_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.img_frame.setFixedSize(720, 300)  # Set fixed size for the frame

        # Przycisk do pokazywania heatmapy
        self.show_heatmap_btn = QPushButton("Pokaż heatmapę")
        self.show_heatmap_btn.setFixedWidth(200)
        self.show_heatmap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_heatmap_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a1b9a;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #8e24aa;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        self.show_heatmap_btn.hide()  # Ukryj na start

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
        
        self.batch_button = QPushButton("Batch test")
        self.batch_button.setFixedWidth(200)
        self.batch_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_button.clicked.connect(self.batch_test)
        self.batch_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #ffa726;
            }
            QPushButton:pressed {
                background-color: #f57c00;
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
        btn_layout.addWidget(self.batch_button)  # Dodaj batch button
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.history_button)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.show_heatmap_btn)  # Dodaj przycisk heatmapy
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
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Image(s)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_paths:
            return
        if len(file_paths) == 1:
            self.process_single_image(file_paths[0])
        else:
            self.process_batch_images(file_paths)

    def process_single_image(self, file_path):
        image = Image.open(file_path).convert('RGB')
        self.original_image = image
        orig_pixmap = self.pil2pixmap(image)
        self.img_label.setPixmap(orig_pixmap.scaled(
            self.img_label.width(), self.img_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        image_224 = image.resize((224, 224), Image.BILINEAR)
        try:
            result = predict_image(image_224)
            self.heatmap_img = result.get('heatmap')
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self._heatmap_visible = False
            if self.heatmap_img is not None:
                self.show_heatmap_btn.show()
            else:
                self.show_heatmap_btn.hide()
            self.current_prediction_id = self.db.save_prediction(
                image_path=file_path,
                brand=result['brand'],
                confidence=result['confidence']
            )
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
            self.good_pred_btn.show()
            self.bad_pred_btn.show()
        except ValueError as e:
            self.current_prediction_id = None
            self.good_pred_btn.hide()
            self.bad_pred_btn.hide()
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.show_heatmap_btn.hide()
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
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.show_heatmap_btn.hide()
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

    def process_batch_images(self, file_paths):
        results = {}
        for file_path in file_paths:
            try:
                image = Image.open(file_path).convert('RGB')
                image_224 = image.resize((224, 224), Image.BILINEAR)
                result = predict_image(image_224)
                results[os.path.basename(file_path)] = {
                    'brand': result.get('brand', 'Błąd'),
                    'confidence': result.get('confidence', 0)
                }
                self.db.save_prediction(
                    image_path=file_path,
                    brand=result.get('brand', 'Błąd'),
                    confidence=result.get('confidence', 0)
                )
            except Exception as e:
                results[os.path.basename(file_path)] = {'brand': f"Błąd: {e}", 'confidence': 0}
        dlg = BatchResultDialog(results, self)
        dlg.exec_()

    def show_heatmap(self):
        """Przełącz widoczność heatmapy"""
        if not hasattr(self, '_heatmap_visible'):
            self._heatmap_visible = False
        if self.heatmap_img is not None:
            if not self._heatmap_visible:
                heatmap_pixmap = self.pil2pixmap(self.heatmap_img)
                # Rozciągnij heatmapę dokładnie do rozmiaru labela (może zniekształcać)
                self.heatmap_label.setPixmap(heatmap_pixmap.scaled(
                    self.heatmap_label.width(), self.heatmap_label.height(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                self.heatmap_label.show()
                self._heatmap_visible = True
                self.show_heatmap_btn.setText("Ukryj heatmapę")
            else:
                self.heatmap_label.clear()
                self.heatmap_label.hide()
                self._heatmap_visible = False
                self.show_heatmap_btn.setText("Pokaż heatmapę")
        else:
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self._heatmap_visible = False
            self.show_heatmap_btn.setText("Pokaż heatmapę")

    def pil2pixmap(self, im):
        """Konwertuje PIL.Image do QPixmap (bez zmiany kolorów i proporcji)"""
        if im.mode != "RGB":
            im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        w, h = im.size
        # Popraw stride (bytesPerLine)
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

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

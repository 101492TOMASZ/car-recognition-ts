import sys
import os
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QFrame, QMessageBox, QHBoxLayout, QSizePolicy,
    QTableWidget, QTableWidgetItem, QDialog, QHeaderView,
    QMenu, QGraphicsOpacityEffect
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QPropertyAnimation, QPoint, QThread, pyqtSignal, QTimer
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
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.current_prediction_id = None
        self.original_image = None
        self.heatmap_img = None
        self._orig_pixmap = None
        self._heatmap_pixmap = None
        self._heatmap_visible = False
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
                padding: 12px;
                margin: 10px 0px 30px 0px;
            }
        """)

    def _make_button(self, text, style='primary', fixed_width=None, fixed_size=None, font_size=14):
        btn = QPushButton(text)
        if fixed_size:
            btn.setFixedSize(*fixed_size)
        if fixed_width:
            btn.setFixedWidth(fixed_width)
        btn.setCursor(Qt.PointingHandCursor)

        styles = {
            'primary': {'bg': '#0078d4', 'hover': '#106ebe', 'pressed': '#005a9e', 'radius': '6px', 'color': 'white', 'padding': '12px 28px'},
            'purple': {'bg': '#6a1b9a', 'hover': '#8e24aa', 'pressed': '#4a148c', 'radius': '6px', 'color': 'white', 'padding': '12px 28px'},
            'good': {'bg': '#2e7d32', 'hover': '#388e3c', 'pressed': '#1b5e20', 'radius': '6px', 'color': 'white', 'padding': '12px 28px'},
            'bad': {'bg': '#c62828', 'hover': '#d32f2f', 'pressed': '#8e0000', 'radius': '6px', 'color': 'white', 'padding': '12px 28px'},
            'icon': {'bg': '#252526', 'hover': '#333337', 'pressed': '#1f1f1f', 'radius': '8px', 'color': '#ffffff', 'padding': '0px'},
            'icon_small': {'bg': '#252526', 'hover': '#3e3e3e', 'pressed': '#1f1f1f', 'radius': '18px', 'color': '#ffffff', 'padding': '0px'},
        }

        s = styles.get(style, styles['primary'])
        qss = f"""
QPushButton {{
    background-color: {s['bg']};
    border: none;
    border-radius: {s['radius']};
    color: {s['color']};
    padding: {s['padding']};
    font-size: {font_size}px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {s['hover']};
}}
QPushButton:pressed {{
    background-color: {s['pressed']};
}}
"""
        btn.setStyleSheet(qss)
        return btn

    def init_ui(self):
        # window sizing and flags
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Car Recognition")
        self.setMinimumSize(880, 980)
        self.setMaximumSize(1920, 1600)

        # top controls (menu, minimize, close)
        self.menu_btn = self._make_button("☰", style='icon', fixed_size=(40, 40))
        self.menu_btn.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.menu_btn.clicked.connect(self.show_menu)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(0)
        top_bar.addWidget(self.menu_btn)
        top_bar.addStretch()
        minimize_btn = self._make_button("–", style='icon_small', fixed_size=(36, 36))
        minimize_btn.clicked.connect(self.showMinimized)
        top_bar.addWidget(minimize_btn)
        close_btn = self._make_button("✕", style='icon_small', fixed_size=(36, 36))
        close_btn.clicked.connect(self.close_window)
        top_bar.addWidget(close_btn)

        # header
        self.header = QLabel("Rozpoznawanie marki samochodu")
        self.header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc = QLabel("Wybierz zdjęcie samochodu, aby rozpoznać jego markę")
        self.desc.setFont(QFont("Segoe UI", 12))
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc.setStyleSheet("color: #cccccc; margin: 10px 0px 20px 0px;")

        # image preview
        self.img_frame = QFrame()
        self.img_frame.setObjectName("preview_frame")
        self.img_frame.setContentsMargins(0, 0, 0, 20)
        img_layout = QHBoxLayout(self.img_frame)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        img_layout.addWidget(self.img_label)
        self.img_frame.setFixedSize(720, 420)

        # buttons
        self.show_heatmap_btn = self._make_button("Pokaż heatmapę", style='purple', fixed_width=200)
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        self.show_heatmap_btn.hide()
        self.button = self._make_button("Wybierz zdjęcie", style='primary', fixed_width=220)
        self.button.clicked.connect(self.choose_file)

        # result area - reserve vertical space
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(110)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_label.setStyleSheet("""
            background-color: #181818;
            color: #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            font-size: 18px;
            font-weight: bold;
        """)

        # layout assembly
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 72)
        layout.setSpacing(22)
        layout.addLayout(top_bar)
        layout.addWidget(self.header)
        layout.addWidget(self.desc)

        preview_container = QVBoxLayout()
        preview_container.setSpacing(14)
        preview_container.setContentsMargins(0, 0, 0, 0)
        preview_container.addWidget(self.img_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        preview_container.addSpacing(28)
        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 12, 0, 0)
        btns_layout.setSpacing(20)
        btns_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btns_layout.addWidget(self.button)
        btns_layout.addWidget(self.show_heatmap_btn)
        preview_container.addLayout(btns_layout)

        layout.addLayout(preview_container)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # window dragging state
        self._drag_active = False
        self._drag_position = None

    def close_window(self):
        """Slot for the close button."""
        self.close()

    # ----------------- image & prediction logic -----------------
    def choose_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz zdjęcie(-a)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_paths:
            return
        if len(file_paths) == 1:
            self.process_single_image(file_paths[0])
        else:
            self.process_batch_images(file_paths)

    def batch_test(self):
        """Open a file dialog to pick multiple images and run batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz zdjęcia do batch testu", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_paths:
            return
        self.process_batch_images(file_paths)

    def process_single_image(self, file_path):
        try:
            image = Image.open(file_path).convert('RGB')
        except Exception as e:
            self.show_error(f"Nie można otworzyć obrazu: {e}")
            return
        self.original_image = image
        self._orig_pixmap = self.pil2pixmap(image)
        self._heatmap_pixmap = None
        self._heatmap_visible = False
        self.show_heatmap_btn.setText("Pokaż heatmapę")
        QTimer.singleShot(0, self.update_display_pixmap)

        try:
            result = predict_image(self.original_image)
            self.heatmap_img = result.get('heatmap')
            if self.heatmap_img is not None:
                # ensure heatmap matches original image size
                self.heatmap_img = self.heatmap_img.resize(self.original_image.size, Image.BILINEAR)
                base = self.original_image.convert("RGBA")
                heatmap_rgba = self.heatmap_img.convert("RGBA")
                blended = Image.blend(base, heatmap_rgba, alpha=0.5)
                self._heatmap_pixmap = self.pil2pixmap(blended)
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
            QTimer.singleShot(250, lambda: self.ask_for_feedback(self.current_prediction_id, result['brand'], result['confidence']))
        except ValueError as e:
            self.current_prediction_id = None
            self._heatmap_visible = False
            self.show_heatmap_btn.hide()
            self.show_heatmap_btn.setText("Pokaż heatmapę")
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
            self._heatmap_visible = False
            self.show_heatmap_btn.hide()
            self.show_heatmap_btn.setText("Pokaż heatmapę")
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
                result = predict_image(image)
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
        if self.original_image is None:
            self._heatmap_visible = False
            self.show_heatmap_btn.setText("Pokaż heatmapę")
            return
        if not self._heatmap_visible:
            if self._heatmap_pixmap is None and self.heatmap_img is not None:
                base = self.original_image.convert("RGBA")
                heatmap_rgba = self.heatmap_img.convert("RGBA")
                blended = Image.blend(base, heatmap_rgba, alpha=0.5)
                self._heatmap_pixmap = self.pil2pixmap(blended)
            self._heatmap_visible = True
            self.show_heatmap_btn.setText("Ukryj heatmapę")
        else:
            self._heatmap_visible = False
            self.show_heatmap_btn.setText("Pokaż heatmapę")
        self.update_display_pixmap()

    def pil2pixmap(self, im):
        if im.mode != "RGB":
            im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        w, h = im.size
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def update_display_pixmap(self):
        target_w = max(1, self.img_label.width())
        target_h = max(1, self.img_label.height())
        if self._heatmap_visible and self._heatmap_pixmap is not None:
            pix = self._heatmap_pixmap.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_label.setPixmap(pix)
        elif self._orig_pixmap is not None:
            pix = self._orig_pixmap.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update_display_pixmap)

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

    def ask_for_feedback(self, pred_id, brand, confidence):
        if pred_id is None:
            return
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ocena predykcji")
        dlg.setText(f"Czy predykcja \"{brand}\" o pewności {confidence:.2f}% jest poprawna?")
        yes_btn = dlg.addButton("Tak, poprawna", QMessageBox.YesRole)
        no_btn = dlg.addButton("Nie, błędna", QMessageBox.NoRole)
        dlg.setIcon(QMessageBox.Question)
        dlg.exec_()
        clicked = dlg.clickedButton()
        if clicked == yes_btn:
            self.mark_prediction(False)
        else:
            self.mark_prediction(True)

    def show_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #252526;
                color: #fff;
                border-radius: 8px;
                font-size: 15px;
                padding: 8px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: #fff;
            }
        """)
        menu.addAction("Wybierz zdjęcie", self.choose_file)
        menu.addSeparator()
        menu.addAction("Batch test", self.batch_test)
        menu.addSeparator()
        menu.addAction("Pokaż historię", self.show_history)
        menu.addSeparator()
        menu.addAction("Autor")
        menu.addSeparator()
        menu.addAction("Zamknij", self.close_window)
        try:
            btn_bottom = self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height()))
            menu.popup(btn_bottom)
        except Exception:
            menu.popup(self.menu_btn.mapToGlobal(self.menu_btn.rect().bottomLeft()))

    def fade_in_widget(self, widget, duration=400):
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.start()
        widget._fade_anim = anim


class ModelLoader(QThread):
    finished = pyqtSignal(object)
    def run(self):
        from predict import load_model  # lazy import
        m = load_model()
        self.finished.emit(m)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CarRecognitionApp()
    ex.show()
    sys.exit(app.exec_())

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QWidget, QSizePolicy,
    QPushButton, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import pandas as pd
import io
from PIL import Image
import os
from datetime import datetime
import sqlite3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Image as RLImage, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from typing import Any
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab.rl_config

class HistoryViewer(QDialog):
    def __init__(self, predictions, database):
        """
        Each prediction expected in the following tuple structure:
        (id, timestamp, image_path, brand, confidence, is_bad_prediction)
        """
        super().__init__()
        self.setWindowTitle("Historia predykcji")
        self.setMinimumSize(900, 600)
        self.predictions = predictions
        self.db = database
        self.init_ui()
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QListWidget {
                background-color: #252526;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QListWidget::item {
                background-color: #2d2d2d;
                border-radius: 4px;
                margin: 5px;
                padding: 10px;
            }
            QListWidget::item:hover {
                background-color: #3e3e3e;
            }
            QComboBox {
                background-color: #3b3b3b;
                border: none;
                border-radius: 4px;
                padding: 8px;
                color: white;
            }
            QComboBox:hover {
                background-color: #4e4e4e;
            }
            QLabel {
                color: #ffffff;
            }
        """)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Sorting options
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sortuj według:")
        self.sort_combobox = QComboBox()
        self.sort_combobox.addItems([
            "Data (najnowsze)", 
            "Data (najstarsze)", 
            "Marka", 
            "Pewność",
            "Jakość predykcji"
        ])
        self.sort_combobox.currentIndexChanged.connect(self.populate_list)
        
        # Export button
        self.export_btn = QPushButton("Eksportuj zaznaczone")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        self.export_btn.clicked.connect(self.export_selected)
        
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combobox)
        sort_layout.addStretch()
        sort_layout.addWidget(self.export_btn)
        layout.addLayout(sort_layout)

        # List widget for predictions
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.list_widget)

        # Increase item height
        self.list_widget.setStyleSheet("""
            QListWidget::item {
                min-height: 150px;
                background-color: #2d2d2d;
                border-radius: 4px;
                margin: 5px;
                padding: 10px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #3e3e3e;
            }
        """)

        self.populate_list()

    def format_confidence(self, conf):
        """Format confidence value as percentage with 2 decimal places"""
        try:
            return f"{float(conf):.2f}%" if conf is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"

    def get_img_path(self, pred_id):
        # Always use the same pattern as database.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(script_dir, "temp", "img")
        img_path = os.path.join(img_dir, f"prediction_{pred_id}.jpg")
        if not os.path.exists(img_path):
            # Ensure directory exists
            os.makedirs(img_dir, exist_ok=True)
            # Try to regenerate from database
            try:
                db_path = os.path.join(script_dir, "temp", "predictions.db")
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    row = cursor.execute("SELECT image_data FROM predictions WHERE id=?", (pred_id,)).fetchone()
                    if row and row[0]:
                        with open(img_path, 'wb') as f:
                            f.write(row[0])
            except Exception as e:
                print(f"Could not regenerate image for prediction {pred_id}: {e}")
        return img_path

    def populate_list(self):
        self.list_widget.clear()

        # Decide sort order
        sort_by = self.sort_combobox.currentText()
        def get_tuple_value(p, idx, default=None):
            return p[idx] if isinstance(p, (tuple, list)) and len(p) > idx else default
            
        if sort_by == "Data (najnowsze)":
            sorted_preds = sorted(self.predictions, key=lambda p: get_tuple_value(p, 1, ''), reverse=True)
        elif sort_by == "Data (najstarsze)":
            sorted_preds = sorted(self.predictions, key=lambda p: get_tuple_value(p, 1, ''))
        elif sort_by == "Marka":
            sorted_preds = sorted(self.predictions, key=lambda p: get_tuple_value(p, 3, '').lower())
        elif sort_by == "Pewność":
            sorted_preds = sorted(self.predictions, key=lambda p: get_tuple_value(p, 4, 0), reverse=True)
        elif sort_by == "Jakość predykcji":
            sorted_preds = sorted(self.predictions, key=lambda p: (get_tuple_value(p, 5, False), get_tuple_value(p, 4, 0)), reverse=True)
        else:
            sorted_preds = self.predictions

        for pred in sorted_preds:
            try:
                id_, timestamp, img_path, brand, conf, is_bad = pred
            except ValueError as e:
                print(f"Error unpacking prediction: {e}")
                continue

            # Always use the correct image path
            img_path = self.get_img_path(id_)

            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 5, 5, 5)
            item_layout.setSpacing(10)

            # Image container with fixed size
            image_container = QWidget()
            image_container.setFixedSize(120, 120)
            image_container.setStyleSheet("background-color: #1e1e1e; border-radius: 4px;")
            image_container_layout = QVBoxLayout(image_container)
            image_container_layout.setContentsMargins(0, 0, 0, 0)

            # Image preview
            image_label = QLabel()
            image_label.setFixedSize(100, 100)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    100, 100,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("Brak obrazu")
            image_container_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)

            # Info container with fixed width
            info_container = QWidget()
            info_container.setFixedWidth(500)
            info_container.setStyleSheet("background-color: #252526; border-radius: 4px; padding: 10px;")
            info_layout = QVBoxLayout(info_container)
            info_layout.setSpacing(5)            # Format date
            date_str = str(timestamp)
            try:
                if isinstance(timestamp, str):
                    date_str = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
                elif isinstance(timestamp, datetime):
                    date_str = timestamp.strftime('%Y-%m-%d %H:%M')
            except Exception:
                pass

            # Format confidence
            try:
                if isinstance(conf, (int, float)):
                    conf_str = f"{conf:.1f}%"

                else:
                    conf_str = "N/A"
            except (ValueError, TypeError):
                conf_str = "N/A"

            # Status with icon
            status_str = "❌ Zła predykcja" if is_bad else "✓ Dobra predykcja"
            status_color = "#c62828" if is_bad else "#2e7d32"            # Create labels for each piece of information
            date_label = QLabel(f"Data: {date_str}")
            brand_label = QLabel(f"Marka: {str(brand)}")
            conf_label = QLabel(f"Pewność: {conf_str}")
            status_label = QLabel(status_str)
            
            # Style labels
            for label in [date_label, brand_label, conf_label]:
                label.setStyleSheet("color: #e0e0e0; font-size: 14px;")
            status_label.setStyleSheet(f"color: {status_color}; font-size: 14px; font-weight: bold;")

            # Add labels to info layout
            info_layout.addWidget(date_label)
            info_layout.addWidget(brand_label)
            info_layout.addWidget(conf_label)
            info_layout.addWidget(status_label)

            # Add containers to main item layout
            item_layout.addWidget(image_container)
            item_layout.addWidget(info_container)
            item_layout.addStretch()

            list_item = QListWidgetItem(self.list_widget)
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, item_widget)

    def export_selected(self):
        """Export the currently selected predictions to a PDF file as a table."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Brak zaznaczenia", "Proszę zaznaczyć przynajmniej jeden element do eksportu.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz jako", "", 
            "Pliki PDF (*.pdf);;Wszystkie pliki (*)",
            options=QFileDialog.Options()
        )
        
        if not file_path:
            return

        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'

        try:
            # Try to register a sensible cross-platform font.
            # On Linux prefer DejaVu Sans (good Unicode coverage). If not found,
            # try several common fallback TTFs. If none are available, fall back
            # to ReportLab's built-in Helvetica which doesn't require a TTF file.
            import platform
            font_name = None
            try:
                if platform.system() == "Windows":
                    arial_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf')
                    if os.path.exists(arial_path):
                        pdfmetrics.registerFont(TTFont("Arial", arial_path))
                        font_name = "Arial"
                else:
                    candidates = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                    ]
                    found = None
                    for p in candidates:
                        if os.path.exists(p):
                            found = p
                            break
                    if found:
                        font_name = "AppSans"
                        pdfmetrics.registerFont(TTFont(font_name, found))
                        reportlab.rl_config.TTFSearchPath.append(os.path.dirname(found))
            except Exception as fe:
                # If font registration fails, we'll fall back to a built-in font
                print(f"Font registration failed: {fe}")

            if not font_name:
                # Use a built-in font that always exists in reportlab.
                font_name = "Helvetica"

            # Prepare data for the table
            data: list[list[Any]] = [["ID", "Data i czas", "Marka", "Pewność (%)", "Jakość", "Obraz"]]
            row_heights = [30]
            image_width = 40 * mm
            image_height = 30 * mm
            styles = getSampleStyleSheet()
            styles["Normal"].fontName = font_name
            styles["Title"].fontName = font_name
            for item in selected_items:
                index = self.list_widget.row(item)
                if index < 0 or index >= len(self.predictions):
                    continue
                pred = self.predictions[index]
                id_, timestamp, img_path, brand, conf, is_bad = pred
                img_path = self.get_img_path(id_)
                try:
                    date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
                except:
                    date = str(timestamp)
                try:
                    conf_str = f"{float(conf):.1f}%" if conf is not None else "N/A"
                except (ValueError, TypeError):
                    conf_str = "N/A"
                status_str = "❌ Zła predykcja" if is_bad else "✓ Dobra predykcja"
                if os.path.exists(img_path):
                    rl_img = RLImage(img_path, width=image_width, height=image_height)
                else:
                    rl_img = Paragraph("Brak obrazu", styles["Normal"])
                data.append([str(id_), date, str(brand), conf_str, status_str, rl_img])
                row_heights.append(int(image_height))
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            table = Table(data, colWidths=[20*mm, 35*mm, 35*mm, 25*mm, 35*mm, image_width], rowHeights=row_heights)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0078d4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements = [Paragraph("Historia predykcji", styles["Title"]), Spacer(1, 12), table]
            doc.build(elements)
            QMessageBox.information(self, "Sukces", "Dane zostały wyeksportowane do PDF pomyślnie.")
        except Exception as e:
            QMessageBox.critical(self, "Błąd eksportu", f"Nie udało się wyeksportować danych: {e}")
            print(f"Export error: {e}")
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QWidget, QSizePolicy,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt  # lub PySide6, zależnie czego używasz

import io
from PIL import Image
import os

class HistoryViewer(QDialog):
    def __init__(self, predictions):
        """
        Each prediction expected in the following tuple structure:
        (id, timestamp, image_path, image_data, brand, confidence)
        """
        super().__init__()
        self.setWindowTitle("Historia predykcji")
        self.setMinimumSize(800, 600)
        self.predictions = predictions
        self.init_ui()

    def convert_pil_to_pixmap(self, img_data):
        """Convert raw image data to QPixmap"""
        img_bytes = io.BytesIO(img_data)
        img = Image.open(img_bytes)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(img_byte_arr.getvalue())
        return pixmap

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Sorting options
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sortuj według:")
        self.sort_combobox = QComboBox()
        self.sort_combobox.addItems(["Marka", "Pewność"])
        self.sort_combobox.currentIndexChanged.connect(self.populate_list)
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combobox)
        sort_layout.addStretch()
        layout.addLayout(sort_layout)

        # List widget for predictions
        self.list_widget = QListWidget()
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.list_widget)

        self.populate_list()

    def populate_list(self):
        self.list_widget.clear()

        # Decide sort order
        sort_by = self.sort_combobox.currentText()
        if sort_by == "Marka":
            sorted_preds = sorted(self.predictions, key=lambda p: p[4].lower() if isinstance(p, (list, tuple)) and len(p) > 4 and isinstance(p[4], str) else "")
        elif sort_by == "Pewność":
            sorted_preds = sorted(self.predictions, key=lambda p: p[5] if isinstance(p, (list, tuple)) and len(p) > 5 and isinstance(p[5], (int, float)) else 0, reverse=True)
        else:
            sorted_preds = self.predictions

        # For each prediction, create a custom item widget
        for pred in sorted_preds:
            # Unpack expected data: (id, timestamp, image_path, image_data, brand, confidence)
            try:
                id_, timestamp, img_path, img_data, brand, conf = pred
            except Exception as e:
                continue

            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 5, 5, 5)
            item_layout.setSpacing(10)

            # Image preview
            pixmap = self.convert_pil_to_pixmap(img_data)
            image_label = QLabel()
            image_label.setFixedSize(100, 100)
            scaled_pixmap = pixmap.scaled(
                image_label.width(), image_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Text information
            info_text = (f"Marka: {brand}\n"
                         f"Pewność: {conf:.2f}%\n"
                         )
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)




            
            item_layout.addWidget(image_label)
            item_layout.addWidget(info_label)
            item_layout.addStretch()

            list_item = QListWidgetItem(self.list_widget)
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, item_widget)
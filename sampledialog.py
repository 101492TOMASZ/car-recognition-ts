import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton, QMessageBox
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


class SampleDialog(QDialog):
    """Okno wyboru przykładowych obrazów z katalogów samples/ lub tools/samples/.

    Użytkownik może wybrać obraz (double-click lub przycisk Otwórz), a wybrana
    ścieżka będzie dostępna w atrybucie `selected_path`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tryb testowy — przykładowe obrazy")
        self.resize(720, 520)
        self.selected_path = None

        lay = QVBoxLayout(self)
        info = QLabel("Wybierz przykładowy obraz z listy poniżej (katalog: samples/ lub tools/samples/)")
        lay.addWidget(info)

        self.listw = QListWidget()
        self.listw.setViewMode(QListWidget.IconMode)
        self.listw.setIconSize(QPixmap(160, 120).size())
        self.listw.setResizeMode(QListWidget.Adjust)
        self.listw.setSpacing(10)
        self.listw.itemDoubleClicked.connect(self._on_open)
        lay.addWidget(self.listw)

        btns = QHBoxLayout()
        self.open_btn = QPushButton("Otwórz")
        self.open_btn.clicked.connect(self._on_open)
        self.cancel_btn = QPushButton("Anuluj")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.cancel_btn)
        lay.addLayout(btns)

        self._load_samples()

    def _load_samples(self):
        root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root, 'samples'),
            os.path.join(root, 'tools', 'samples'),
        ]
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        paths = []
        for d in candidates:
            if os.path.isdir(d):
                for name in sorted(os.listdir(d)):
                    p = os.path.join(d, name)
                    if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                        paths.append(p)

        if not paths:
            QMessageBox.information(self, 'Brak próbek', 'Dodaj obrazy do katalogu samples/ w katalogu projektu.')
            return

        for p in paths:
            try:
                pix = QPixmap(p).scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                it = QListWidgetItem(os.path.basename(p))
                it.setIcon(QIcon(pix))
                it.setData(Qt.UserRole, p)
                self.listw.addItem(it)
            except Exception:
                continue

    def _on_open(self, *args):
        it = self.listw.currentItem()
        if not it:
            QMessageBox.information(self, 'Wybór', 'Zaznacz obraz na liście.')
            return
        self.selected_path = it.data(Qt.UserRole)
        self.accept()

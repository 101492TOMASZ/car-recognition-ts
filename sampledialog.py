import os
from utils_paths import samples_dirs
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton, QMessageBox,
    QAbstractItemView, QListView
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QEvent


class SampleDialog(QDialog):
    """Okno wyboru przykładowych obrazów z katalogów samples/ lub tools/samples/.

    - Multiwybór (przeciąganie gumką, Ctrl/Shift).
    - Tryb live: po puszczeniu LPM:
      * 1 zaznaczony: emituje imageSelected(str)
      * >1 zaznaczonych: emituje imagesSelected(list[str])
    """

    imageSelected = pyqtSignal(str)
    imagesSelected = pyqtSignal(list)

    def __init__(self, parent=None, live_mode: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Tryb testowy — przykładowe obrazy")
        self.resize(720, 520)
        self.selected_path = None
        self.live_mode = bool(live_mode)

        lay = QVBoxLayout(self)
        info = QLabel(
            "Kliknij obraz, aby uruchomić predykcję.\n"
            "Zaznacz wiele obrazów przeciągając myszą lub używając Ctrl/Shift.\n"
        )
        lay.addWidget(info)

        self.listw = QListWidget()
        self.listw.setViewMode(QListWidget.IconMode)
        self.listw.setIconSize(QPixmap(160, 120).size())
        self.listw.setResizeMode(QListWidget.Adjust)
        self.listw.setSpacing(10)

        # Multiwybór i gumka
        self.listw.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listw.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.listw.setMovement(QListView.Static)
        self.listw.setSelectionRectVisible(True)

        # W trybie live emitujemy po puszczeniu przycisku myszy
        if self.live_mode:
            self.listw.viewport().installEventFilter(self)
        else:
            # poza live można użyć double-click do otwarcia pojedynczego elementu
            self.listw.itemDoubleClicked.connect(self._on_open)

        lay.addWidget(self.listw)

        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("Zamknij")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        lay.addLayout(btns)

        self._load_samples()

    def eventFilter(self, obj, event):
        if obj is self.listw.viewport() and self.live_mode:
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._emit_selection()
        return super().eventFilter(obj, event)

    def _emit_selection(self):
        items = self.listw.selectedItems()
        if not items:
            return
        if len(items) == 1:
            p = items[0].data(Qt.UserRole)
            if p:
                self.selected_path = p
                self.imageSelected.emit(p)
        else:
            paths = [it.data(Qt.UserRole) for it in items if it.data(Qt.UserRole)]
            if paths:
                self.selected_path = None
                self.imagesSelected.emit(paths)

    def _load_samples(self):
        candidates = [str(p) for p in samples_dirs()]
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
        # Używane tylko poza live_mode (double click)
        it = self.listw.currentItem()
        if not it:
            QMessageBox.information(self, 'Wybór', 'Zaznacz obraz na liście.')
            return
        self.selected_path = it.data(Qt.UserRole)
        if self.selected_path:
            self.imageSelected.emit(self.selected_path)

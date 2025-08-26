import os
import sys
import time
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QPushButton, QHBoxLayout, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from database import get_all_records, export_record_pdf, get_record


class HistoryViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Historia predykcji')
        self.resize(900, 600)

        layout = QVBoxLayout()

        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.MultiSelection)
        self.listw.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.listw)

        bottom = QHBoxLayout()

        self.preview = QLabel('Wybierz rekord')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(300, 200)
        bottom.addWidget(self.preview)

        btns = QVBoxLayout()
        self.export_btn = QPushButton('Eksportuj zaznaczone do PDF')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_pdf)
        btns.addWidget(self.export_btn)

        self.export_all_btn = QPushButton('Eksportuj wszystkie do PDF')
        self.export_all_btn.clicked.connect(self._export_all_pdf)
        btns.addWidget(self.export_all_btn)

        btns.addStretch(1)
        bottom.addLayout(btns)

        layout.addLayout(bottom)
        self.setLayout(layout)

        self._load()

    def _load(self):
        self.listw.clear()
        recs = get_all_records()
        for r in recs:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['timestamp']))
            label = f"{r['id']}: {r.get('predicted')} ({r.get('confidence'):.2f}%) - {t}"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, r['id'])
            self.listw.addItem(it)

    def _on_select(self):
        items = self.listw.selectedItems()
        if not items:
            self.preview.setText('Wybierz rekord')
            self.export_btn.setEnabled(False)
            return
        rid = items[0].data(Qt.UserRole)
        rec = get_record(rid)
        img = rec.get('saved_image') or rec.get('image_path')
        if img and os.path.isfile(img):
            pix = QPixmap(img)
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setText('Brak obrazu')
        self.export_btn.setEnabled(True)

    def _export_pdf(self):
        items = self.listw.selectedItems()
        if not items:
            return
        rid = items[0].data(Qt.UserRole)
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', f'record_{rid}.pdf', 'PDF files (*.pdf)')
        if not out:
            return
        try:
            export_record_pdf(rid, out)
            QMessageBox.information(self, 'OK', f'Zapisano {out}')
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można wyeksportować: {e}')

    def _export_all_pdf(self):
        # export all records currently in the list
        all_ids = [self.listw.item(i).data(Qt.UserRole) for i in range(self.listw.count())]
        if not all_ids:
            QMessageBox.information(self, 'Info', 'Brak rekordów do eksportu')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', f'records_all.pdf', 'PDF files (*.pdf)')
        if not out:
            return
        try:
            from database import export_records_pdf
            export_records_pdf(all_ids, out)
            QMessageBox.information(self, 'OK', f'Zapisano {out}')
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można wyeksportować: {e}')

import os
import time
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
from database import get_all_records, export_record_pdf, get_record, export_records_pdf


class HistoryViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Historia predykcji')
        self.resize(960, 620)
        self.setStyleSheet(
            """
            QDialog { background: #f6f8fb; color: #1f2328; }
            QListWidget { background: #ffffff; border: 1px solid #e6e9ee; border-radius: 10px; padding: 6px; }
            QListWidget::item { padding: 8px; margin: 2px 0; }
            QListWidget::item:selected { background: #e7f2ff; border-radius: 6px; }
            QLabel#Preview { background: #ffffff; border: 1px solid #e6e9ee; border-radius: 10px; }
            QLabel#Stats { background: #ffffff; border: 1px solid #e6e9ee; border-radius: 10px; padding: 10px; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #38e4ae); color: #ffffff; border: none; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5b95ff, stop:1 #48e9b6); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d7df2, stop:1 #2fd09f); }
            QPushButton:disabled { background: #c7cdd7; color: #f2f4f7; }
            """
        )

        main_layout = QHBoxLayout(self)

        # left: list with checkboxes
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.SingleSelection)
        self.listw.itemSelectionChanged.connect(self._on_select)
        self.listw.itemClicked.connect(self._on_item_clicked)
        self.listw.setMinimumWidth(380)
        main_layout.addWidget(self.listw)

        # right: preview + stats + buttons
        right = QVBoxLayout()

        self.preview = QLabel('Wybierz rekord')
        self.preview.setObjectName('Preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setFixedSize(480, 360)
        right.addWidget(self.preview, alignment=Qt.AlignTop)

        self.stats = QLabel('')
        self.stats.setObjectName('Stats')
        self.stats.setWordWrap(True)
        right.addWidget(self.stats)

        # buttons row
        btns = QHBoxLayout()
        self.btn_export_selected = QPushButton('Eksportuj zaznaczony')
        self.btn_export_selected.clicked.connect(self._export_pdf)
        self.btn_export_selected.setEnabled(False)
        self.btn_export_checked = QPushButton('Eksportuj wybrane (checkbox)')
        self.btn_export_checked.clicked.connect(self._export_checked_pdf)
        self.btn_export_all = QPushButton('Eksportuj wszystkie')
        self.btn_export_all.clicked.connect(self._export_all_pdf)
        self.btn_refresh = QPushButton('Odśwież')
        self.btn_refresh.clicked.connect(self._load)
        for b in (self.btn_export_selected, self.btn_export_checked, self.btn_export_all, self.btn_refresh):
            btns.addWidget(b)
        btns.addItem(QSpacerItem(6, 6))

        right.addLayout(btns)
        main_layout.addLayout(right)

        self._load()

    def _load(self):
        self.listw.clear()
        recs = get_all_records()
        for r in recs:
            try:
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(r.get('timestamp') or 0)))
            except Exception:
                t = ''
            try:
                conf = float(r.get('confidence') or 0.0)
            except Exception:
                conf = 0.0
            label = f"#{r['id']} • {r.get('predicted')} ({conf:.2f}%) • {t}"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, r['id'])
            # make item checkable for export selection
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Unchecked)
            # lightly color incorrect records
            try:
                if not bool(r.get('correct')):
                    it.setBackground(QColor(255, 220, 220))
            except Exception:
                pass
            self.listw.addItem(it)
        self.preview.setText('Wybierz rekord')
        self.preview.setPixmap(QPixmap())
        self.stats.setText('')
        self.btn_export_selected.setEnabled(False)

    def _on_select(self):
        items = self.listw.selectedItems()
        if not items:
            self.preview.setText('Wybierz rekord')
            self.preview.setPixmap(QPixmap())
            self.stats.setText('')
            self.btn_export_selected.setEnabled(False)
            return
        rid = items[0].data(Qt.UserRole)
        rec = get_record(rid)
        if not rec:
            self.preview.setText('Brak danych')
            self.preview.setPixmap(QPixmap())
            self.stats.setText('')
            self.btn_export_selected.setEnabled(False)
            return
        img = rec.get('saved_image') or rec.get('image_path')
        if img and os.path.isfile(img):
            pix = QPixmap(img)
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.setText('Brak obrazu')
        # show stats for the selected record
        try:
            conf = float(rec.get('confidence') or 0.0)
        except Exception:
            conf = 0.0
        try:
            pt = float(rec.get('processing_time') or 0.0)
        except Exception:
            pt = 0.0
        self.stats.setText(
            f"ID: {rec.get('id')}\n"
            f"Predicted: {rec.get('predicted')}\n"
            f"Confidence: {conf:.2f}%\n"
            f"Correct: {bool(rec.get('correct'))}\n"
            f"Processing time: {pt:.3f}s\n"
            f"Timestamp: {rec.get('timestamp')}"
        )
        self.btn_export_selected.setEnabled(True)

    def _export_pdf(self):
        items = self.listw.selectedItems()
        if not items:
            QMessageBox.information(self, 'Eksport', 'Najpierw wybierz rekord.')
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

    def _on_item_clicked(self, item):
        # checkbox / click updates preview and stats
        try:
            rid = item.data(Qt.UserRole)
            rec = get_record(rid)
            if not rec:
                return
            img = rec.get('saved_image') or rec.get('image_path')
            if img and os.path.isfile(img):
                pix = QPixmap(img)
                self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.preview.setText('Brak obrazu')
            try:
                conf = float(rec.get('confidence') or 0.0)
            except Exception:
                conf = 0.0
            try:
                pt = float(rec.get('processing_time') or 0.0)
            except Exception:
                pt = 0.0
            self.stats.setText(
                f"ID: {rec.get('id')}\n"
                f"Predicted: {rec.get('predicted')}\n"
                f"Confidence: {conf:.2f}%\n"
                f"Correct: {bool(rec.get('correct'))}\n"
                f"Processing time: {pt:.3f}s\n"
                f"Timestamp: {rec.get('timestamp')}"
            )
            self.btn_export_selected.setEnabled(True)
        except Exception:
            pass

    def _export_all_pdf(self):
        ids = [self.listw.item(i).data(Qt.UserRole) for i in range(self.listw.count())]
        if not ids:
            QMessageBox.information(self, 'Eksport', 'Brak rekordów.')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', f'history.pdf', 'PDF files (*.pdf)')
        if not out:
            return
        try:
            export_records_pdf(ids, out)
            QMessageBox.information(self, 'Eksport', f'Zapisano: {out}')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd eksportu', str(e))

    def _export_checked_pdf(self):
        ids = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == Qt.Checked:
                ids.append(it.data(Qt.UserRole))
        if not ids:
            QMessageBox.information(self, 'Eksport', 'Zaznacz rekordy (checkbox).')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', f'selected.pdf', 'PDF files (*.pdf)')
        if not out:
            return
        try:
            export_records_pdf(ids, out)
            QMessageBox.information(self, 'Eksport', f'Zapisano: {out}')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd eksportu', str(e))

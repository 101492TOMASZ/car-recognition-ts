import os
import sys
import time
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from database import get_all_records, export_record_pdf, get_record
from PyQt5.QtGui import QColor


class HistoryViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Historia predykcji')
        self.resize(900, 600)
        main_layout = QHBoxLayout()

        # left: list with checkboxes
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.SingleSelection)
        self.listw.itemSelectionChanged.connect(self._on_select)
        self.listw.itemClicked.connect(self._on_item_clicked)
        self.listw.setMinimumWidth(360)
        main_layout.addWidget(self.listw)

        # right: preview + stats + buttons
        right = QVBoxLayout()

        self.preview = QLabel('Wybierz rekord')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setFixedSize(420, 320)
        right.addWidget(self.preview, alignment=Qt.AlignTop)

        self.stats_label = QLabel('')
        self.stats_label.setAlignment(Qt.AlignTop)
        right.addWidget(self.stats_label)

        btns = QHBoxLayout()
        self.export_checked_btn = QPushButton('Eksportuj zaznaczone (checkbox) do PDF')
        self.export_checked_btn.clicked.connect(self._export_checked_pdf)
        btns.addWidget(self.export_checked_btn)

        self.export_selected_btn = QPushButton('Eksportuj zaznaczony rekord do PDF')
        self.export_selected_btn.setEnabled(False)
        self.export_selected_btn.clicked.connect(self._export_pdf)
        btns.addWidget(self.export_selected_btn)

        self.export_all_btn = QPushButton('Eksportuj wszystkie do PDF')
        self.export_all_btn.clicked.connect(self._export_all_pdf)
        btns.addWidget(self.export_all_btn)

        right.addLayout(btns)

        main_layout.addLayout(right)
        self.setLayout(main_layout)

        self._load()

    def _load(self):
        self.listw.clear()
        recs = get_all_records()
        for r in recs:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['timestamp']))
            label = f"{r['id']}: {r.get('predicted')} ({r.get('confidence'):.2f}%) - {t}"
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
        # show stats for the selected record
        try:
            s = []
            s.append(f"ID: {rec.get('id')}")
            s.append(f"Predicted: {rec.get('predicted')}")
            s.append(f"Confidence: {rec.get('confidence'):.2f}%")
            s.append(f"Correct: {bool(rec.get('correct'))}")
            s.append(f"Processing time: {rec.get('processing_time'):.3f}s")
            import datetime
            s.append(f"Timestamp: {datetime.datetime.fromtimestamp(rec.get('timestamp')).isoformat()}")
            self.stats_label.setText('\n'.join(s))
        except Exception:
            self.stats_label.setText('Brak danych')
        self.export_selected_btn.setEnabled(True)

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

    def _on_item_clicked(self, item):
        # when an item is clicked (either checked or selected), show its details
        try:
            rid = item.data(Qt.UserRole)
            rec = get_record(rid)
            if rec:
                img = rec.get('saved_image') or rec.get('image_path')
                if img and os.path.isfile(img):
                    pix = QPixmap(img)
                    self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.preview.setText('Brak obrazu')
                # update stats
                try:
                    s = []
                    s.append(f"ID: {rec.get('id')}")
                    s.append(f"Predicted: {rec.get('predicted')}")
                    s.append(f"Confidence: {rec.get('confidence'):.2f}%")
                    s.append(f"Correct: {bool(rec.get('correct'))}")
                    s.append(f"Processing time: {rec.get('processing_time'):.3f}s")
                    import datetime
                    s.append(f"Timestamp: {datetime.datetime.fromtimestamp(rec.get('timestamp')).isoformat()}")
                    self.stats_label.setText('\n'.join(s))
                except Exception:
                    self.stats_label.setText('Brak danych')
                self.export_selected_btn.setEnabled(True)
        except Exception:
            pass

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

    def _export_checked_pdf(self):
        # export only checked items
        checked_ids = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            try:
                if it.checkState() == Qt.Checked:
                    checked_ids.append(it.data(Qt.UserRole))
            except Exception:
                continue
        if not checked_ids:
            QMessageBox.information(self, 'Info', 'Brak zaznaczonych rekordów do eksportu')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', f'records_checked.pdf', 'PDF files (*.pdf)')
        if not out:
            return
        try:
            from database import export_records_pdf
            export_records_pdf(checked_ids, out)
            QMessageBox.information(self, 'OK', f'Zapisano {out}')
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można wyeksportować: {e}')

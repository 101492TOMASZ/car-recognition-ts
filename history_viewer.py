import os
import time
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
from database import get_all_records, get_record, export_records_table_pdf, delete_records


class HistoryViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Historia predykcji')
        self.resize(960, 620)
        # zachowaj referencję do motywu (domyślnie dark jeśli brak w rodzicu)
        self.theme = getattr(parent, 'theme', 'dark')

        main_layout = QHBoxLayout(self)

        # left: list (multi-select)
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.ExtendedSelection)
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
        self.btn_export = QPushButton('Eksportuj zaznaczone')
        self.btn_export.clicked.connect(self._export_selected_pdf)
        self.btn_export.setEnabled(False)
        self.btn_delete = QPushButton('Usuń zaznaczone')
        self.btn_delete.setObjectName('DangerBtn')
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setEnabled(False)
        self.btn_refresh = QPushButton('Odśwież')
        self.btn_refresh.clicked.connect(self._load)
        for b in (self.btn_export, self.btn_delete, self.btn_refresh):
            btns.addWidget(b)
        btns.addItem(QSpacerItem(6, 6))

        right.addLayout(btns)
        main_layout.addLayout(right)

        self._load()
        # zastosuj motyw po zbudowaniu UI
        self.apply_theme(self.theme)

    # ---------------- THEME (spójny z GUI.py) -----------------
    def apply_theme(self, mode: str):
        dark = (mode == 'dark')
        if dark:
            bg_gradient = "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e1115, stop:1 #1b2733)"
            shell_bg = "rgba(255,255,255,0.06)"
            panel_bg = "rgba(255,255,255,0.05)"
            border_col = "rgba(255,255,255,0.10)"
            text_col = "#edf2f7"
            accent = "#0ea5e9"
            accent_hover = "#0d8fd0"
            accent_down = "#0b78b2"
            subtle = "#334155"
            list_sel = "rgba(14,165,233,0.22)"
        else:
            bg_gradient = "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f3f5f7, stop:1 #e3e8ec)"
            shell_bg = "rgba(255,255,255,0.72)"
            panel_bg = "rgba(255,255,255,0.55)"
            border_col = "rgba(0,0,0,0.10)"
            text_col = "#1b2834"
            accent = "#0077ff"
            accent_hover = "#0065d6"
            accent_down = "#0052ad"
            subtle = "#c6d2dc"
            list_sel = "rgba(0,119,255,0.18)"

        style = f"""
            QDialog {{
                background: {bg_gradient};
                color: {text_col};
                font-family: 'Segoe UI','Arial';
                font-size: 14px;
            }}
            QListWidget {{
                background: {panel_bg};
                border: 1px solid {border_col};
                border-radius: 18px; padding: 8px; outline: 0; selection-background-color: {list_sel};
            }}
            QListWidget::item {{ padding: 8px 10px; margin: 2px 0; border-radius: 10px; }}
            QListWidget::item:selected {{ background: {list_sel}; color: {text_col}; }}
            QLabel#Preview, QLabel#Stats {{
                background: {panel_bg};
                border: 1px solid {border_col};
                border-radius: 18px; padding: 10px;
            }}
            QPushButton {{
                background: {accent};
                color: {'#f1f5f9' if dark else '#ffffff'};
                border: 0px solid transparent; border-radius: 14px;
                padding: 8px 16px; font-size: 13px; font-weight: 600; margin: 4px 6px;
            }}
            QPushButton:hover {{ background: {accent_hover}; }}
            QPushButton:pressed {{ background: {accent_down}; }}
            QPushButton:disabled {{ background: {subtle}; color: {'#64748b' if dark else '#94a3b8'}; }}
            QPushButton#DangerBtn {{
                background: {'#ef4444' if dark else '#dc2626'};
                color: #ffffff;
            }}
            QPushButton#DangerBtn:hover {{ background: {'#dc2626' if dark else '#b91c1c'}; }}
            QPushButton#DangerBtn:pressed {{ background: {'#b91c1c' if dark else '#991b1b'}; }}
        """
        try:
            self.setStyleSheet(style)
            self.theme = mode
        except Exception:
            pass

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
            it.setFlags(it.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            try:
                if not bool(r.get('correct')):
                    it.setBackground(QColor(255, 220, 220))
            except Exception:
                pass
            self.listw.addItem(it)
        self.preview.setText('Wybierz rekord')
        self.preview.setPixmap(QPixmap())
        self.stats.setText('')
        self.btn_export.setEnabled(False)
        self.btn_delete.setEnabled(False)

    def _on_select(self):
        items = self.listw.selectedItems()
        if not items:
            self.preview.setText('Wybierz rekord')
            self.preview.setPixmap(QPixmap())
            self.stats.setText('')
            self.btn_export.setEnabled(False)
            self.btn_delete.setEnabled(False)
            return
        rid = items[0].data(Qt.UserRole)
        rec = get_record(rid)
        if not rec:
            self.preview.setText('Brak danych')
            self.preview.setPixmap(QPixmap())
            self.stats.setText('')
            self.btn_export.setEnabled(False)
            self.btn_delete.setEnabled(False)
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
        has_sel = len(items) > 0
        self.btn_export.setEnabled(has_sel)
        self.btn_delete.setEnabled(has_sel)

    def _on_item_clicked(self, item):
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
            self.btn_export.setEnabled(True)
            self.btn_delete.setEnabled(True)
        except Exception:
            pass

    def _delete_selected(self):
        items = self.listw.selectedItems()
        if not items:
            QMessageBox.information(self, 'Usuń', 'Zaznacz co najmniej jeden rekord (Ctrl/Cmd + klik).')
            return
        ids = [it.data(Qt.UserRole) for it in items]
        msg = f"Czy na pewno usunąć {len(ids)} rekord(ów)? Operacja jest nieodwracalna."
        reply = QMessageBox.question(self, 'Potwierdź usunięcie', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            deleted = delete_records(ids)
            self._load()
            QMessageBox.information(self, 'Usunięto', f'Usunięto: {deleted} rekord(ów).')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd usuwania', str(e))

    def _export_selected_pdf(self):
        items = self.listw.selectedItems()
        if not items:
            QMessageBox.information(self, 'Eksport', 'Zaznacz co najmniej jeden rekord (Ctrl/Cmd + klik).')
            return
        ids = [it.data(Qt.UserRole) for it in items]
        default_name = 'selected.pdf' if len(ids) > 1 else f'record_{ids[0]}.pdf'
        out, _ = QFileDialog.getSaveFileName(self, 'Zapisz PDF', default_name, 'PDF files (*.pdf)')
        if not out:
            return
        try:
            export_records_table_pdf(ids, out)
            QMessageBox.information(self, 'Eksport', f'Zapisano: {out}')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd eksportu', str(e))

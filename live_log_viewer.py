import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import QTimer

class LiveLogViewer(QDialog):
    """Modeless live tail viewer for application log file."""
    def __init__(self, log_path: str, parent=None, interval_ms: int = 1000, max_initial=1500, max_lines=4000):
        super().__init__(parent)
        self.setWindowTitle("Logi (na żywo)")
        self.resize(780, 460)
        self.log_path = log_path
        self.follow = True
        self.last_size = 0
        self.max_lines = max_lines
        self.max_initial = max_initial

        lay = QVBoxLayout(self)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet("QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; font-size:12px; }")
        lay.addWidget(self.txt)

        btn_row = QHBoxLayout()
        self.clear_btn = QPushButton("Wyczyść")
        close_btn = QPushButton("Zamknij")
        btn_row.addStretch(1)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        
        self.clear_btn.clicked.connect(self._clear_view)
        close_btn.clicked.connect(self.close)

        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._initial_load()

    def _initial_load(self):
        try:
            if not os.path.isfile(self.log_path):
                self.txt.setPlainText(f"Brak pliku logów: {self.log_path}")
                return
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                data = f.read()
                self.last_size = f.tell()
            lines = data.splitlines()
            if len(lines) > self.max_initial:
                lines = lines[-self.max_initial:]
            self.txt.setPlainText("\n".join(lines))
            QTimer.singleShot(0, self._scroll_bottom)
        except Exception as e:
            self.txt.setPlainText(f"Błąd wczytania logów: {e}")

    def _poll(self):
        if not self.follow:
            return
        try:
            if not os.path.isfile(self.log_path):
                return
            size = os.path.getsize(self.log_path)
            if size < self.last_size:
                self.last_size = 0
            if size == self.last_size:
                return
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.last_size)
                chunk = f.read()
                self.last_size = f.tell()
            if not chunk:
                return
            at_bottom = self._is_at_bottom()
            self.txt.moveCursor(self.txt.textCursor().End)
            self.txt.insertPlainText(chunk)
            self._trim_lines()
            if at_bottom:
                self._scroll_bottom()
        except Exception:
            pass

    def _trim_lines(self):
        doc = self.txt.document()
        if doc.blockCount() <= self.max_lines:
            return
        over = doc.blockCount() - self.max_lines
        cursor = self.txt.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(over):
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _is_at_bottom(self):
        sb = self.txt.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _scroll_bottom(self):
        sb = self.txt.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_view(self):
        self.txt.clear()
        QTimer.singleShot(0, self._scroll_bottom)

    def closeEvent(self, e):
        try:
            self.timer.stop()
        except Exception:
            pass
        super().closeEvent(e)

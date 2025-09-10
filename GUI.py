import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QMessageBox,
    QFrame, QSizePolicy, QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEvent, QTimer
from PIL import Image
import torch
from predict import predict_image
from database import init_db, insert_record, save_image_copy
from history_viewer import HistoryViewer
from sampledialog import SampleDialog
import time


class PredictionThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, image_path, yolo_model, classifier, idx_to_label, device='cpu'):
        super().__init__()
        self.image_path = image_path
        self.yolo = yolo_model
        self.classifier = classifier
        self.idx_to_label = idx_to_label
        self.device = device
    def run(self):
        start = time.time()
        res = predict_image(self.image_path, self.yolo, self.classifier, self.idx_to_label, device=self.device)
        res['processing_time'] = time.time() - start
        # Treat 'no_vehicle' as a normal finished state so GUI can show a friendly message
        if res.get('no_vehicle'):
            self.finished.emit(res)
        elif res.get('message'):
            self.error.emit(res['message'])
        else:
            self.finished.emit(res)


class BatchPredictionThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, image_paths, yolo_model, classifier, idx_to_label, device='cpu'):
        super().__init__()
        self.paths = list(image_paths)
        self.yolo = yolo_model
        self.classifier = classifier
        self.idx_to_label = idx_to_label
        self.device = device

    def run(self):
        import time as _time
        results = []
        brand_counts = {}
        total_time = 0.0
        no_vehicle_count = 0

        for p in self.paths:
            start = _time.time()
            try:
                res = predict_image(p, self.yolo, self.classifier, self.idx_to_label, device=self.device)
            except Exception as e:
                res = {'path': p, 'message': f'error: {e}'}
            res['path'] = p
            proc = res.get('processing_time') or (_time.time() - start)
            res['processing_time'] = float(proc)
            total_time += float(proc)

            if res.get('no_vehicle'):
                no_vehicle_count += 1
            else:
                b = res.get('brand')
                if b:
                    brand_counts[b] = brand_counts.get(b, 0) + 1

            results.append(res)

        summary = {
            'count': len(self.paths),
            'no_vehicle': no_vehicle_count,
            'avg_time': (total_time / max(1, len(self.paths))),
            'brand_counts': brand_counts,
            'results': results,
        }
        self.finished.emit(summary)


class CarCropGUI(QWidget):
    def _get_brand_svg_path(self, brand):
        # Map brand names to SVG file paths in the 'logos' directory
        brand_map = {
            'BMW': 'logos/bmw-svgrepo-com.svg',
            'Audi': 'logos/audi-svgrepo-com.svg',
            'Mercedes': 'logos/mercedes-svgrepo-com.svg',
            'Porsche': 'logos/porsche-svgrepo-com.svg',
            'Volkswagen': 'logos/volkswagen-svgrepo-com.svg',
        }
        # Normalize brand name for matching
        key = str(brand).strip().capitalize()
        # Try direct match, then lower-case match
        path = brand_map.get(key)
        if not path:
            for k in brand_map:
                if k.lower() == str(brand).strip().lower():
                    path = brand_map[k]
                    break
        if path and os.path.isfile(path):
            return path
        return None

    def _show_brand_overlay(self, brand):
        """Show a centered semi-transparent SVG logo (watermark style)."""
        svg_path = self._get_brand_svg_path(brand)
        if not svg_path:
            if getattr(self, 'brand_overlay', None):
                self.brand_overlay.hide()
            return
        self._current_brand = brand
        from PyQt5.QtSvg import QSvgWidget
        if getattr(self, 'brand_overlay', None):
            self.brand_overlay.hide()
            self.brand_overlay.setParent(None)
        # Make overlay a child of the image_label so it never paints outside
        # the displayed pixmap area (image_frame has inner margins).
        self.brand_overlay = QSvgWidget(self.image_label)
        target_w = int(self.image_label.width() * 0.55)
        target_h = int(self.image_label.height() * 0.55)
        self.brand_overlay.setFixedSize(max(60, target_w), max(60, target_h))
        self.brand_overlay.load(svg_path)
        self.brand_overlay.setStyleSheet("background: rgba(255,255,255,0.60); border: 1px solid rgba(22,34,46,0.08); border-radius: 20px;")
        self.brand_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.brand_overlay.show()
        self._reposition_brand_overlay()
        try:
            self._fade_in_widget(self.brand_overlay, duration=380)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # No hover-hide logic needed now; overlay is passive.
        return super().eventFilter(obj, event)

    def __init__(self):  # consolidated (removed duplicate earlier definition)
        super().__init__()
        self.setWindowTitle("AutoDentifier")
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(900, 640)

        # Zablokuj maksymalizację (przycisk + akcja WM)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)

        self.setStyleSheet("""
            QWidget { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #111827, stop:1 #1e293b); font-family: 'Segoe UI', 'Arial'; font-size: 14px; color: #e2e8f0; }
            QFrame#ShellFrame { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.07); border-radius: 28px; }
            QFrame#NavPanel { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.07); border-radius: 22px; }
            QLabel#TitleLabel { font-size: 30px; font-weight: 700; color: #f8fafc; margin: 8px 8px 16px 8px; letter-spacing: 1px; }
            QFrame#ImageFrame { background: rgba(255,255,255,0.10); border-radius: 26px; border: 1px solid rgba(255,255,255,0.09); }
            QLabel#ImageLabel { background: #0f172a; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05); }
            QLabel#HeatmapLabel { border-radius: 18px; }
            QLabel#ResultLabel { font-size: 15px; font-weight: 600; color: #0f172a; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 18px; padding: 10px 18px; margin-top: 14px; }
            QPushButton { background: #0ea5e9; color: #f1f5f9; border: none; border-radius: 14px; padding: 8px 16px; font-size: 13px; font-weight: 600; margin: 6px 4px; min-width: 140px; }
            QPushButton:hover { background: #0284c7; }
            QPushButton:pressed { background: #0369a1; }
            QPushButton:disabled { background: #334155; color: #64748b; }
            QPushButton#DangerBtn { background: #dc2626; }
            QPushButton#DangerBtn:hover { background: #b91c1c; }
            QPushButton#SecondaryBtn { background: #6366f1; }
            QPushButton#SecondaryBtn:hover { background: #4f46e5; }
            QTableWidget { background: #1e293b; gridline-color: #334155; selection-background-color: #334155; }
            QHeaderView::section { background: #334155; color: #e2e8f0; border: none; padding: 4px 6px; }
            QMessageBox { background: #1e293b; }
        """)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = None
        self.classifier = None
        self.idx_to_label = None
        self.heatmap_visible = False
        self.feedback_saved = False
        self.brand_overlay = None
        self._build_ui()

        self._animations = []
        # ensure db exists
        try:
            init_db()
        except Exception:
            pass

        # Zablokuj zmianę rozmiaru okna (stała szerokość/wysokość)
        try:
            self.setFixedSize(self.size())  # bazuje na setGeometry powyżej
        except Exception:
            pass

    def _fade_in_widget(self, widget, duration=300):
        """Apply a fade-in animation to a widget (keeps reference to animation)."""
        try:
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(duration)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.start()
            # keep reference until finished
            self._animations.append(anim)

            def _on_finished():
                try:
                    self._animations.remove(anim)
                except ValueError:
                    pass

            anim.finished.connect(_on_finished)
        except Exception:
            pass

    def _build_ui(self):
        # Wrapper shell frame to create glass effect look
        shell = QFrame()
        shell.setObjectName("ShellFrame")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(18, 18, 18, 18)
        shell_layout.setSpacing(22)

        # Navigation panel
        nav = QFrame()
        nav.setObjectName("NavPanel")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(14, 14, 14, 14)
        nav_layout.setSpacing(10)

        self.title_label = QLabel("AutoDentifier")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignHCenter)
        nav_layout.addWidget(self.title_label)

        # Buttons
        self.load_button = QPushButton("Wybierz obraz(y)")
        self.load_button.clicked.connect(self.load_image)
        nav_layout.addWidget(self.load_button)

        self.test_button = QPushButton("Tryb testowy")
        self.test_button.setObjectName("SecondaryBtn")
        self.test_button.clicked.connect(self._open_test_mode)
        nav_layout.addWidget(self.test_button)

        self.history_button = QPushButton("Historia")
        self.history_button.clicked.connect(self._open_history)
        nav_layout.addWidget(self.history_button)

        self.heatmap_button = QPushButton("Pokaż heatmapę")
        self.heatmap_button.setEnabled(False)
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        nav_layout.addWidget(self.heatmap_button)

        self.confirm_button = QPushButton("Zgłoś błąd")
        self.confirm_button.setObjectName("DangerBtn")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setVisible(False)
        self.confirm_button.clicked.connect(self._on_confirm_click)
        nav_layout.addWidget(self.confirm_button)

        nav_layout.addStretch(1)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(12)

        # Image frame & stacked internals (responsive)
        self.image_frame = QFrame()
        self.image_frame.setObjectName("ImageFrame")
        self.image_frame.setMinimumSize(560, 400)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            shadow = QGraphicsDropShadowEffect(self.image_frame)
            shadow.setBlurRadius(32)
            shadow.setXOffset(0)
            shadow.setYOffset(12)
            shadow.setColor(Qt.black)
            self.image_frame.setGraphicsEffect(shadow)
        except Exception:
            pass

        # Layout-managed image label
        from PyQt5.QtWidgets import QVBoxLayout as _QVBox
        img_layout = _QVBox(self.image_frame)
        img_layout.setContentsMargins(18, 18, 18, 18)
        img_layout.setSpacing(0)
        self.image_label = QLabel()
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout.addWidget(self.image_label)
        # Heatmap overlay as child of image_label (absolute positioning within)
        self.heatmap_label = QLabel(self.image_label)
        self.heatmap_label.setObjectName("HeatmapLabel")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.heatmap_label.hide()

        content_layout.addWidget(self.image_frame, alignment=Qt.AlignHCenter)

        self.result_label = QLabel("Gotowy – wybierz obraz")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        shell_layout.addWidget(nav)
        shell_layout.addLayout(content_layout, stretch=1)

        outer = QVBoxLayout()
        outer.setContentsMargins(30, 30, 30, 30)
        outer.addWidget(shell)
        self.setLayout(outer)

    def _reposition_brand_overlay(self):
        """Center SVG overlay within image_label bounds (respects margins)."""
        if not getattr(self, 'brand_overlay', None):
            return
        try:
            frame_w = self.image_label.width()
            frame_h = self.image_label.height()
            if frame_w <= 0 or frame_h <= 0:
                return
            target_w = int(frame_w * 0.55)
            target_h = int(frame_h * 0.55)
            self.brand_overlay.setFixedSize(max(60, target_w), max(60, target_h))
            # Place relative to image_label (parent)
            x = (frame_w - self.brand_overlay.width()) // 2
            y = (frame_h - self.brand_overlay.height()) // 2
            self.brand_overlay.move(x, y)
            self.brand_overlay.raise_()
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            # Ensure heatmap overlay always fills image_label
            self.heatmap_label.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
            if hasattr(self, 'current_image_path') and self.current_image_path:
                pix = QPixmap(self.current_image_path)
                scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            if self.heatmap_visible:
                self._render_heatmap_overlay()
        except Exception:
            pass
        self._reposition_brand_overlay()

    def event(self, e):
        """Handle move/activate to correct any transient painting drift after alt-tab or dragging."""
        et = e.type()
        if et in (QEvent.Move, QEvent.WindowActivate, QEvent.ApplicationActivate):
            # Defer to end of event loop so geometry is final
            QTimer.singleShot(0, self._post_window_adjust)
        return super().event(e)

    def _post_window_adjust(self):
        try:
            if self.heatmap_visible:
                self._render_heatmap_overlay()
            self._reposition_brand_overlay()
        except Exception:
            pass

    def _render_heatmap_overlay(self):
        """Regenerate the centered heatmap overlay onto transparent canvas sized like image_label."""
        if not self.heatmap_data:
            return
        try:
            import base64
            from PyQt5.QtGui import QPixmap, QPainter
            img_bytes = base64.b64decode(self.heatmap_data)
            qimg = QPixmap()
            if not qimg.loadFromData(img_bytes, 'PNG'):
                return
            target = self.image_label.size()
            scaled = qimg.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(target)
            canvas.fill(Qt.transparent)
            p = QPainter(canvas)
            x = (canvas.width() - scaled.width()) // 2
            y = (canvas.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
            p.end()
            self.heatmap_label.setPixmap(canvas)
            self.heatmap_label.raise_()
        except Exception:
            pass

    def _refresh_base_image(self):
        """Rescale and set the original image pixmap (used after hiding heatmap)."""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            return
        try:
            pix = QPixmap(self.current_image_path)
            scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        except Exception:
            pass

    def load_models(self):
        if self.yolo is None:
            try:
                from ultralytics import YOLO
                y = YOLO('yolov8s.pt')
                try:
                    y.to(self.device)
                except Exception:
                    pass
                self.yolo = y
            except Exception:
                self.yolo = None

    def load_image(self):
        # hide report button while predicting
        try:
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
            self.feedback_saved = False
        except Exception:
            pass
        paths, _ = QFileDialog.getOpenFileNames(self, 'Wybierz obraz(y)', '', 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not paths:
            return
        if len(paths) == 1:
            self._set_image_and_predict(paths[0])
            return
        # batch
        self._run_batch(paths)

    def _run_batch(self, paths):
        # UI prep
        try:
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
            # ensure stacked layout on image
            # no stacked layout now
            self.heatmap_button.setEnabled(False)
            self.result_label.setText(f'Batch: przetwarzanie {len(paths)} obrazów...')
        except Exception:
            pass

        # Ensure YOLO model is loaded before batch predictions
        try:
            if self.yolo is None:
                self.result_label.setText('Ładowanie modelu YOLO...')
                QApplication.processEvents()
                self.load_models()
        except Exception:
            pass

        self.batch_thread = BatchPredictionThread(paths, self.yolo, None, None, device=self.device)
        self.batch_thread.finished.connect(self._on_batch_finished)
        self.batch_thread.error.connect(self._on_batch_error)
        self.batch_thread.start()

    def _set_image_and_predict(self, path: str):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd', f'Nie można otworzyć obrazu: {e}')
            return
        self.current_image = img
        self.current_image_path = path
        pix = QPixmap(path)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        # animate smooth fade-in for the newly loaded image
        try:
            self._fade_in_widget(self.image_label, duration=400)
        except Exception:
            pass
        # reset/close any existing heatmap and disable the heatmap button until a new one is generated
        try:
            self.heatmap_label.clear()
            # ensure base visible (heatmap overlays anyway)
            self.heatmap_visible = False
            self.heatmap_button.setEnabled(False)
            self.heatmap_button.setText("Pokaż heatmapę")
        except Exception:
            pass

        self.result_label.setText('Predicting...')

    # start prediction thread
    # Ensure YOLO model is loaded before single prediction
        try:
            if self.yolo is None:
                self.result_label.setText('Ładowanie modelu YOLO...')
                QApplication.processEvents()
                self.load_models()
        except Exception:
            pass
        self.pred_thread = PredictionThread(self.current_image_path, self.yolo, None, None, device=self.device)
        self.pred_thread.finished.connect(self._on_pred_finished)
        self.pred_thread.error.connect(self._on_pred_error)
        self.pred_thread.start()

    def _open_test_mode(self):
        dlg = SampleDialog(self)
        if dlg.exec_() == QDialog.Accepted and dlg.selected_path:
            self._set_image_and_predict(dlg.selected_path)

    def _on_batch_finished(self, summary):
        # Show summary dialog with table and optional CSV export
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle('Wyniki batch testu')
            lay = QVBoxLayout(dlg)

            head = QLabel(
                f"Plików: {summary.get('count', 0)}  |  Bez pojazdu: {summary.get('no_vehicle', 0)}  |  Śr. czas: {summary.get('avg_time', 0.0):.2f}s"
            )
            lay.addWidget(head)

            table = QTableWidget()
            rows = len(summary.get('results', []))
            table.setRowCount(rows)
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(['Plik', 'Marka', 'Pewność (%)', 'Czas (s)', 'Status'])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

            for r, res in enumerate(summary.get('results', [])):
                path = os.path.basename(res.get('path', ''))
                brand = res.get('brand') or ''
                conf = res.get('confidence')
                try:
                    conf_str = f"{float(conf or 0.0):.2f}"
                except Exception:
                    conf_str = ''
                t = res.get('processing_time')
                try:
                    t_str = f"{float(t or 0.0):.2f}"
                except Exception:
                    t_str = ''
                status = 'OK' if not res.get('no_vehicle') else 'Brak pojazdu'
                if res.get('message') and not res.get('no_vehicle'):
                    status = res.get('message')

                table.setItem(r, 0, QTableWidgetItem(path))
                table.setItem(r, 1, QTableWidgetItem(str(brand)))
                table.setItem(r, 2, QTableWidgetItem(conf_str))
                table.setItem(r, 3, QTableWidgetItem(t_str))
                table.setItem(r, 4, QTableWidgetItem(status))

            lay.addWidget(table)

            # Buttons
            btns = QHBoxLayout()
            save_btn = QPushButton('Zapisz CSV...')
            close_btn = QPushButton('Zamknij')
            btns.addStretch(1)
            btns.addWidget(save_btn)
            btns.addWidget(close_btn)
            lay.addLayout(btns)

            def _save_csv():
                import csv
                path, _ = QFileDialog.getSaveFileName(dlg, 'Zapisz wyniki', 'batch_results.csv', 'CSV (*.csv)')
                if not path:
                    return
                try:
                    with open(path, 'w', newline='', encoding='utf-8') as f:
                        w = csv.writer(f)
                        w.writerow(['file', 'brand', 'confidence', 'time_s', 'status'])
                        for res in summary.get('results', []):
                            fn = res.get('path', '')
                            brand = res.get('brand') or ''
                            conf = res.get('confidence') or 0.0
                            t = res.get('processing_time') or 0.0
                            status = 'no_vehicle' if res.get('no_vehicle') else (res.get('message') or 'ok')
                            try:
                                conf = float(conf or 0.0)
                            except Exception:
                                conf = 0.0
                            try:
                                t = float(t or 0.0)
                            except Exception:
                                t = 0.0
                            w.writerow([fn, brand, f"{conf:.2f}", f"{t:.2f}", status])
                    QMessageBox.information(self, 'Zapisano', f'Zapisano: {path}')
                except Exception as e:
                    QMessageBox.warning(self, 'Błąd', f'Nie udało się zapisać CSV: {e}')

            save_btn.clicked.connect(_save_csv)
            close_btn.clicked.connect(dlg.accept)
            dlg.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Batch', f'Nie udało się pokazać wyników: {e}')

        # reset small UI bits
        try:
            self.result_label.setText('Batch zakończony')
        except Exception:
            pass

    def _on_batch_error(self, err):
        QMessageBox.warning(self, 'Batch', f'Błąd batch: {err}')

    def _on_pred_finished(self, res):
        # handle explicit no-vehicle gate
        if res.get('no_vehicle'):
            # disable heatmap, hide report button, don't save anything
            self.heatmap_data = None
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.heatmap_button.setEnabled(False)
            self.heatmap_button.setText("Pokaż heatmapę")
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
            self.last_result = None
            msg = res.get('message') or 'Zdjęcie nie przedstawia pojazdu'
            self.result_label.setText(msg)
            if hasattr(self, 'brand_overlay') and self.brand_overlay:
                self.brand_overlay.hide()
            return
        brand = res.get('brand')
        conf = res.get('confidence')
        proc_time = res.get('processing_time', 0.0)
        self.heatmap_data = res.get('heatmap')
        if self.heatmap_data:
            self.heatmap_button.setEnabled(True)
        else:
            self.heatmap_button.setEnabled(False)
            self.heatmap_label.clear()
        # store last result
        self.last_result = {'brand': brand, 'confidence': conf, 'image_path': self.current_image_path}
        self._show_brand_overlay(brand)
        # auto-save as correct by default (single entry)
        if not self.feedback_saved:
            try:
                entry = {'timestamp': int(time.time()), 'image': self.last_result.get('image_path'), 'predicted': self.last_result.get('brand'), 'confidence': float(self.last_result.get('confidence') or 0.0), 'correct': True}
                hist_path = os.path.join(os.path.dirname(__file__), 'history.jsonl')
                with open(hist_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                self.feedback_saved = True
            except Exception:
                pass
        # show report button
        self.confirm_button.setVisible(True)
        self.confirm_button.setEnabled(True)
        # save a copy of the image and insert into DB
        try:
            # save a copy into the hidden database image folder
            try:
                dst = save_image_copy(self.current_image_path)
            except Exception:
                # fallback: try to save via PIL into the hidden dir
                base = os.path.basename(self.current_image_path)
                dst = save_image_copy(self.current_image_path, prefix='copied')
            rid = insert_record(self.current_image_path, dst, brand, float(conf or 0.0), True, float(proc_time), int(time.time()))
            try:
                # store DB record id to allow marking it incorrect later
                self.last_result['record_id'] = int(rid)
            except Exception:
                pass
        except Exception:
            pass
        if brand is None:
            self.result_label.setText(res.get('message', 'Brak wyników'))
        else:
            try:
                self.result_label.setText(f"Brand={brand}, Confidence={float(conf or 0.0):.2f}%")
            except Exception:
                self.result_label.setText(f"Brand={brand}")

    def toggle_heatmap(self):
        # Use QStackedLayout to swap exactly - prevents layout shift.
        if not self.heatmap_data:
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            if getattr(self, 'brand_overlay', None):
                self.brand_overlay.show(); self._reposition_brand_overlay()
            return
        if self.heatmap_visible:
            # Hiding heatmap overlay
            try:
                self.heatmap_label.hide()
                self.heatmap_label.clear()
            except Exception:
                pass
            self._refresh_base_image()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            if getattr(self, 'brand_overlay', None):
                try:
                    self.brand_overlay.show()
                    self._reposition_brand_overlay()
                    self.brand_overlay.raise_()
                except Exception:
                    pass
            return
        # Show overlay
        self._render_heatmap_overlay()
        try:
            self._fade_in_widget(self.heatmap_label, duration=320)
        except Exception:
            pass
        self.heatmap_label.show()
        self.heatmap_button.setText("Schowaj heatmapę")
        self.heatmap_visible = True
        if getattr(self, 'brand_overlay', None):
            self.brand_overlay.hide()

    def show_heatmap(self):
        self.toggle_heatmap()

    def _on_pred_error(self, err):
        self.result_label.setText(f"Prediction error: {err}")
        try:
            self.confirm_button.setVisible(False)
            self.confirm_button.setEnabled(False)
        except Exception:
            pass

    def _open_history(self):
        try:
            hv = HistoryViewer(self)
            hv.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można otworzyć historii: {e}')

    def _on_confirm_click(self):
        # user reports incorrect prediction
        if not getattr(self, 'last_result', None):
            QMessageBox.information(self, 'Info', 'Brak wyniku do zgłoszenia')
            return
        try:
            entry = {'timestamp': int(time.time()), 'image': self.last_result.get('image_path'), 'predicted': self.last_result.get('brand'), 'confidence': float(self.last_result.get('confidence') or 0.0), 'correct': False, 'reported': True}
            hist_path = os.path.join(os.path.dirname(__file__), 'history.jsonl')
            with open(hist_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            # mark DB record as incorrect if we have it
            try:
                from database import update_record_correct
                rid = self.last_result.get('record_id')
                if rid:
                    update_record_correct(int(rid), False)
            except Exception:
                pass
            QMessageBox.information(self, 'Dziękuję', 'Zgłoszenie zapisane')
            self.confirm_button.setEnabled(False)
            self.confirm_button.setVisible(False)
            self.feedback_saved = True
        except Exception as e:
            QMessageBox.warning(self, 'Błąd', f'Nie można zapisać zgłoszenia: {e}')

    def changeEvent(self, event):
        # Jeśli WM spróbuje zmaksymalizować okno, przywróć normalny stan
        if event.type() == QEvent.WindowStateChange and self.windowState() & Qt.WindowMaximized:
            self.setWindowState(Qt.WindowNoState)
        super().changeEvent(event)

    def closeEvent(self, event):
        # wait for prediction thread to finish or terminate it to avoid crashes
        if getattr(self, 'pred_thread', None) is not None and self.pred_thread.isRunning():
            reply = QMessageBox.question(self, 'Zamykanie', 'Predykcja w toku. Poczekać na zakończenie?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.pred_thread.wait(10000)
            else:
                try:
                    self.pred_thread.terminate()
                except Exception:
                    pass
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CarCropGUI()
    win.show()
    sys.exit(app.exec_())

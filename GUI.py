import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QMessageBox,
    QFrame, QSizePolicy, QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEvent
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Crop & Predict GUI")
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(900, 640)

        # Zablokuj maksymalizację (przycisk + akcja WM)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)

        self.setStyleSheet("""
            QWidget { background: #f5f7fb; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 14px; color: #1f2328; }
            QLabel#TitleLabel { font-size: 24px; font-weight: 700; color: #0f141a; margin: 2px 0 10px 0; }
            QLabel#ResultLabel { font-size: 16px; font-weight: 600; color: #0b72bf; margin: 10px 0 12px 0; background: #ffffff; border: 1px solid #e6e9ee; border-radius: 12px; padding: 10px 14px; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f8cff, stop:1 #38e4ae); color: #ffffff; border: none; border-radius: 12px; padding: 9px 18px; font-size: 14px; font-weight: 600; margin: 6px; min-width: 140px; min-height: 36px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5b95ff, stop:1 #48e9b6); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d7df2, stop:1 #2fd09f); }
            QPushButton:disabled { background: #c7cdd7; color: #f2f4f7; }
            QFrame#ImageFrame { background: #ffffff; border-radius: 14px; border: 1px solid #e6e9ee; padding: 8px; }
            QLabel#HeatmapLabel { border-radius: 10px; }
        """)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = None
        self.classifier = None
        self.idx_to_label = None

        # header
        self.title_label = QLabel("Car Crop & Predict GUI")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # image frame
        self.image_frame = QFrame()
        self.image_frame.setObjectName("ImageFrame")
        self.image_frame.setFixedSize(800, 460)
        self.image_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # subtle shadow under image frame
        try:
            shadow = QGraphicsDropShadowEffect(self.image_frame)
            shadow.setBlurRadius(18)
            shadow.setXOffset(0)
            shadow.setYOffset(6)
            shadow.setColor(Qt.black)
            self.image_frame.setGraphicsEffect(shadow)
        except Exception:
            pass

        self.image_label = QLabel(self.image_frame)
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(784, 444)
        self.image_label.move(8, 8)

        self.heatmap_label = QLabel(self.image_frame)
        self.heatmap_label.setObjectName("HeatmapLabel")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setFixedSize(self.image_label.size())
        self.heatmap_label.move(8, 8)
        self.heatmap_label.hide()

        # result label
        self.result_label = QLabel("")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)

        # buttons
        self.load_button = QPushButton("Wybierz obraz")
        self.load_button.clicked.connect(self.load_image)
        self.test_button = QPushButton("Tryb testowy")
        self.test_button.clicked.connect(self._open_test_mode)
        self.heatmap_button = QPushButton("Pokaż heatmapę")
        self.heatmap_button.setEnabled(False)
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        self.confirm_button = QPushButton("Zgłoś nieprawidłową predykcję")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setVisible(False)
        self.confirm_button.clicked.connect(self._on_confirm_click)
        self.history_button = QPushButton("Historia")
        self.history_button.clicked.connect(self._open_history)

        # layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.image_frame, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(12)
        btns_layout.addStretch(1)
        btns_layout.addWidget(self.load_button)
        btns_layout.addWidget(self.test_button)
        btns_layout.addWidget(self.heatmap_button)
        btns_layout.addWidget(self.confirm_button)
        btns_layout.addWidget(self.history_button)
        btns_layout.addStretch(1)
        main_layout.addLayout(btns_layout)

        self.setLayout(main_layout)

        # internal state
        self.current_image = None
        self.current_image_path = None
        self.pred_thread = None
        self.last_result = None
        self.heatmap_data = None
        self.heatmap_visible = False
        self.feedback_saved = False
        self.load_models()
        self.confirm_button.setVisible(False)
        self.confirm_button.setEnabled(False)
        # container for running animations so they are not garbage-collected
        self._animations = []
        # ensure db exists
        try:
            init_db()
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

        paths, _ = QFileDialog.getOpenFileNames(self, 'Wybierz obraz(y)', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
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
            self.heatmap_label.hide()
            self.heatmap_button.setEnabled(False)
            self.result_label.setText(f'Batch: przetwarzanie {len(paths)} obrazów...')
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
            self.heatmap_label.hide()
            self.heatmap_visible = False
            self.heatmap_button.setEnabled(False)
            self.heatmap_button.setText("Pokaż heatmapę")
        except Exception:
            pass

        self.result_label.setText('Predicting...')

        # start prediction thread
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
        if not self.heatmap_data:
            self.heatmap_label.clear()
            self.heatmap_label.hide()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            return
        if self.heatmap_visible:
            self.heatmap_label.hide()
            self.heatmap_button.setText("Pokaż heatmapę")
            self.heatmap_visible = False
            return
        import base64
        from PyQt5.QtGui import QPixmap
        try:
            img_bytes = base64.b64decode(self.heatmap_data)
            qimg = QPixmap()
            qimg.loadFromData(img_bytes, 'PNG')
            target = self.image_label.size()
            scaled = qimg.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(target)
            canvas.fill(Qt.transparent)
            from PyQt5.QtGui import QPainter
            painter = QPainter(canvas)
            x = (canvas.width() - scaled.width()) // 2
            y = (canvas.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
            self.heatmap_label.setPixmap(canvas)
            # set initial invisible and animate fade-in
            self.heatmap_label.setVisible(True)
            try:
                self._fade_in_widget(self.heatmap_label, duration=350)
            except Exception:
                pass
            self.heatmap_button.setText("Schowaj heatmapę")
            self.heatmap_visible = True
        except Exception as e:
            print(f"show heatmap error: {e}")

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

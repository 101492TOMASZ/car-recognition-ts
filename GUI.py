import sys
import os
import json
import yaml
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QMessageBox,
    QFrame, QSizePolicy, QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpacerItem, QPlainTextEdit, QToolButton, QStyle
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEvent, QTimer, QSize
from PIL import Image
import torch
from predict import predict_image
from database import init_db, insert_record, save_image_copy
from logger_setup import setup_logger
from prediction_service import PredictionService
from history_viewer import HistoryViewer
from sampledialog import SampleDialog
from live_log_viewer import LiveLogViewer
import time



class PredictionThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, image_path: str, service: PredictionService):
        super().__init__()
        self.image_path = image_path
        self.service = service
    def run(self):
        import time as _t
        start = _t.time()
        try:
            res = self.service.predict(self.image_path)
        except Exception as e:
            self.error.emit(str(e))
            return
        res['processing_time'] = _t.time() - start
        if res.get('no_vehicle'):
            self.finished.emit(res)
        elif res.get('message'):
            self.error.emit(res['message'])
        else:
            self.finished.emit(res)


class BatchPredictionThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, image_paths, service: PredictionService):
        super().__init__()
        self.paths = list(image_paths)
        self.service = service

    def run(self):
        import time as _time
        results = []
        brand_counts = {}
        total_time = 0.0
        no_vehicle_count = 0
        total = len(self.paths)
        for idx, p in enumerate(self.paths, start=1):
            start = _time.time()
            try:
                res = self.service.predict(p)
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
            self.progress.emit(idx, total)
        summary = {
            'count': total,
            'no_vehicle': no_vehicle_count,
            'avg_time': (total_time / max(1, total)),
            'brand_counts': brand_counts,
            'results': results,
        }
        self.finished.emit(summary)




class CarCropGUI(QWidget):
    # Minimalny sugerowany rozmiar obszaru obrazu – używany jako minimum, nie sztywna blokada
    IMAGE_AREA_MIN = (880, 600)
    IMAGE_ASPECT = 760 / 520  # referencyjne proporcje (szer/wys)
    IMAGE_WIDTH_RATIO = 0.82  # zwiększony maksymalny udział szerokości
    IMAGE_TARGET_AREA_RATIO = 0.55  # ~55% powierzchni okna dla większego obrazu
    RESULT_MAX_HEIGHT = 170  # ogranicz wysokość okienka z wynikiem
    BRAND_LOGO_MAX_RATIO = 0.55  # maksymalnie 55% szerokości / wysokości obszaru obrazu
    BRAND_LOGO_SCALE_OVERRIDES = {  # indywidualne korekty (mnożniki) dla konkretnych marek
        'Porsche': 0.48,  # trochę mniejsze żeby cały kształt mieścił się w białym kwadracie
    }
    # Wbudowane czyste SVG (bez atrybutów width/height) – ładowane z pamięci
    BRAND_SVG = {
        'Audi': (
            """<svg viewBox='0 0 32 32' fill='none' xmlns='http://www.w3.org/2000/svg'>
<path fill-rule='evenodd' clip-rule='evenodd' d='M5.58991 11C2.5245 11 -0.0510283 13.5306 0.000767908 16.5905C0.0516066 19.5934 2.54027 22 5.59027 22C6.8999 22 8.10608 21.5563 9.06016 20.812C10.0143 21.5562 11.2205 22 12.5302 22C13.8398 22 15.046 21.5563 16.0001 20.812C16.9542 21.5562 18.1604 22 19.4701 22C20.7798 22 21.986 21.5563 22.94 20.812C23.8942 21.5562 25.1004 22 26.4101 22C29.4597 22 31.9484 19.5938 31.9992 16.5905C32.051 13.5304 29.4754 11 26.4099 11C25.1111 11 23.9003 11.4542 22.94 12.2104C21.9795 11.4542 20.7686 11 19.4698 11C18.171 11 16.9603 11.4542 15.9999 12.2104C15.0396 11.4542 13.8288 11 12.53 11C11.2312 11 10.0204 11.4542 9.06004 12.2105C8.09965 11.4542 6.88874 11 5.58991 11ZM12.5302 20.7518C11.5705 20.7518 10.6833 20.4441 9.96546 19.9232C10.7071 19.0067 11.1581 17.8512 11.1794 16.5905C11.2016 15.2766 10.7395 14.0604 9.96228 13.1002C10.686 12.5665 11.578 12.2482 12.53 12.2482C13.4821 12.2482 14.374 12.5665 15.0977 13.1001C14.3204 14.0604 13.8582 15.2768 13.8803 16.5906C13.9018 17.8512 14.3529 19.0068 15.0947 19.9233C14.3769 20.4442 13.4898 20.7518 12.5302 20.7518ZM9.91349 16.5697C9.92973 15.6102 9.60767 14.7149 9.06006 13.9908C8.51245 14.7149 8.19039 15.6102 8.20663 16.5697C8.22228 17.4943 8.5373 18.3464 9.06013 19.0368C9.58286 18.3464 9.89783 17.4944 9.91349 16.5697ZM6.94071 16.5905C6.91847 15.2766 7.38057 14.0604 8.15783 13.1002C7.43411 12.5665 6.54203 12.2482 5.58991 12.2482C3.2229 12.2482 1.22685 14.2162 1.26669 16.5697C1.30595 18.8891 3.2289 20.7518 5.59027 20.7518C6.54983 20.7518 7.43703 20.4442 8.15483 19.9232C7.41304 19.0067 6.96205 17.8512 6.94071 16.5905ZM18.1193 16.5905C18.1416 15.2766 17.6795 14.0604 16.9022 13.1001C17.6258 12.5665 18.5178 12.2482 19.4698 12.2482C20.4219 12.2482 21.314 12.5665 22.0378 13.1002C21.2605 14.0604 20.7984 15.2766 20.8206 16.5905C20.8419 17.8512 21.2929 19.0067 22.0347 19.9232C21.3169 20.4442 20.4297 20.7518 19.4701 20.7518C18.5105 20.7518 17.6233 20.4441 16.9054 19.9232C17.6471 19.0067 18.098 17.8512 18.1193 16.5905ZM22.0865 16.5697C22.0703 15.6102 22.3924 14.7148 22.94 13.9907C23.4877 14.7148 23.8097 15.6102 23.7934 16.5697C23.7777 17.4944 23.4627 18.3464 22.94 19.0368C22.4172 18.3464 22.1022 17.4943 22.0865 16.5697ZM26.4101 20.7518C25.4504 20.7518 24.5632 20.4441 23.8453 19.9232C24.587 19.0068 25.0379 17.8513 25.0593 16.5906C25.0817 15.2765 24.6196 14.0603 23.8423 13.1001C24.5659 12.5665 25.4579 12.2482 26.4099 12.2482C28.7771 12.2482 30.7732 14.216 30.7333 16.5697C30.694 18.8895 28.7711 20.7518 26.4101 20.7518ZM16.8534 16.599C16.8696 15.6395 16.5476 14.7442 15.9999 14.0201C15.4523 14.7442 15.1303 15.6395 15.1465 16.599C15.1622 17.5236 15.4772 18.3757 16 19.0661C16.5227 18.3757 16.8377 17.5237 16.8534 16.599Z' fill='#000000'/>
</svg>"""
        ),
        'BMW': (
            """<svg fill='#000000' version='1.1' id='Layer_2' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 271 274' xml:space='preserve'>
<path id='Shape' d='M135.5,17.5C69.5,17.5,16,71,16,137s53.5,119.5,119.5,119.5S255,203,255,137 S201.5,17.5,135.5,17.5L135.5,17.5z M247.2,137c0,61.7-50,111.7-111.7,111.7S23.8,198.7,23.8,137s50-111.7,111.7-111.7 S247.2,75.3,247.2,137z'/> 
<g id='Group' transform='translate(7.304985, 3.108504)'>
<path id='W_60_' d='M186.1,83.2c1.6,1.7,4,4.6,5.3,6.3l24.2-15.4c-1.2-1.6-3.1-4-4.6-5.7l-15.3,10.1l-1,0.9 l0.8-1.1l6.8-13.5l-4.8-4.8l-13.5,6.8l-1.1,0.8l0.9-1l10.1-15.3c-1.8-1.5-3.5-2.9-5.7-4.6l-15.4,24.2c1.9,1.5,4.5,3.7,6.1,5.2 l14.5-7.5l0.9-0.7l-0.7,0.9L186.1,83.2z'/> 
<path id='M_60_' d='M131.2,52.6l6.6-14.8l0.4-1.3l-0.1,1.4l0.7,19.8c2.3,0.2,4.7,0.5,7.1,0.9l-1.1-29.3 c-3.3-0.4-6.6-0.6-9.9-0.8l-6.5,16.2l-0.2,1.2l-0.2-1.2l-6.5-16.2c-3.3,0.1-6.6,0.4-9.9,0.8l-1.1,29.3c2.4-0.4,4.8-0.7,7.1-0.9 l0.7-19.8l-0.1-1.4l0.4,1.3l6.6,14.8H131.2z'/> 
<path id='B_x5F_22d_60_' d='M77.7,75.9c3.8-4,6-8.7,2.2-13.1c-2.1-2.4-5.6-2.9-8.5-1.7l-0.3,0.1l0.1-0.3 c0.4-1.1,0.7-4.8-2.4-7.3c-1.5-1.2-3.4-1.7-5.3-1.5c-3.6,0.4-6.3,2.8-13.9,11.2c-2.3,2.5-5.6,6.5-7.6,9.1L62.7,92 C69.6,84.4,72.3,81.6,77.7,75.9z M50.8,71.1c4.2-5.1,8.6-9.7,10.6-11.5c0.6-0.6,1.3-1.2,2.2-1.4c1.4-0.4,2.8,0.6,3.1,2 c0.3,1.4-0.6,2.7-1.5,3.8C62.9,66.5,54.8,75,54.8,75L50.8,71.1z M58.9,78.8c0,0,7.9-8.3,10.4-11c1-1.1,1.7-1.7,2.4-2 c0.9-0.4,1.9-0.5,2.8,0.1c0.9,0.6,1.3,1.6,1.1,2.6c-0.3,1.2-1.2,2.3-2,3.2c-1.1,1.2-10.4,11.1-10.4,11.1L58.9,78.8z'/> 
</g>
<path d='M135.5,66.2V137H64.7C64.7,97.8,96.3,66.2,135.5,66.2z'/>
<path d='M206.3,137c0,39.2-31.7,70.8-70.8,70.8V137H206.3z'/>
</svg>"""
        ),
        'Porsche': (
            """<svg width="800px" height="800px" viewBox="0 0 192.756 192.756" xmlns="http://www.w3.org/2000/svg">
<g fill-rule="evenodd" clip-rule="evenodd">

<path fill="#ffffff" d="M0 0h192.756v192.756H0V0z"/>

<path d="M52.347 98.48H36.653c-.24 0-.478-.159-.718-.478v-3.116c0-.478.237-.718.718-.718h15.694c.078.082.195.121.357.121.24.162.36.36.36.598v3.116l-.717.477zm-24.558-3.112h-16.65v-1.199h16.65l.722.481-.722.718zm50.188-.241c-.079.163-.199.241-.358.241H61.091v-1.199h16.527c.159 0 .358.163.598.481-.081.159-.162.318-.239.477zm-47.073-1.198c0-1.196-.601-1.797-1.796-1.797H8.504v8.507h2.635v-3.354h17.969c1.196 0 1.796-.638 1.796-1.917v-1.439zm24.435 0c0-1.196-.601-1.797-1.796-1.797H35.216c-1.199 0-1.797.601-1.797 1.797v4.672c0 1.358.598 2.038 1.797 2.038h18.327c1.195 0 1.796-.68 1.796-2.038v-4.672zm25.513 4.194c0-.48-.241-.958-.718-1.439.477-.237.718-.676.718-1.316v-1.439c0-1.196-.641-1.797-1.917-1.797H58.573v8.507h2.519v-3.354h16.287c.078.077.237.217.478.419.24.201.361.38.361.536v2.398h2.635v-2.515h-.001zm25.156-.838c0-1.277-.641-1.917-1.916-1.917H86.84c-.159 0-.4-.237-.718-.718l.298-.299a.581.581 0 0 1 .42-.182h19.167v-2.037H85.524c-1.199 0-1.797.601-1.797 1.797v1.439c0 1.28.598 1.917 1.797 1.917h17.247c.479 0 .719.24.719.718-.078.081-.18.182-.299.299a.576.576 0 0 1-.42.179H83.728v2.158h20.364c1.275 0 1.916-.68 1.916-2.038v-1.316zm25.152 1.195h-19.043l-.361-.478v-3.116c0-.237.119-.478.361-.718h19.043v-2.037h-20.121c-1.281 0-1.918.601-1.918 1.797V98.6c0 1.358.637 2.038 1.918 2.038h20.121V98.48zm24.676-6.349h-2.395v3.236h-17.252v-3.236h-2.514v8.507h2.514v-3.354h17.252v3.354h2.395v-8.507zm25.393 6.349h-19.762v-1.195h19.762v-1.917h-19.762v-1.199h19.762v-2.037h-22.396v8.507h22.396V98.48zM183.066 92.118c-.668 0-1.186.519-1.186 1.185s.518 1.186 1.186 1.186c.67 0 1.186-.52 1.186-1.186s-.516-1.185-1.186-1.185zm0 .169c.574 0 1.002.45 1.002 1.017 0 .568-.428 1.016-1.002 1.016s-1-.448-1-1.016c0-.568.426-1.017 1-1.017zm-.259 1.093h.283l.385.607h.197l-.408-.607c.186-.033.346-.141.346-.383 0-.246-.139-.379-.447-.379h-.525v1.369h.17v-.607h-.001zm0-.144v-.474h.309c.166 0 .324.045.324.236 0 .233-.203.238-.41.238h-.223z"/>

</g>

</svg>>"""
        ),
        'Volkswagen': (
            """<svg fill='#000000' viewBox='0 0 24 24' role='img' xmlns='http://www.w3.org/2000/svg'><path d='M12 0C5.36 0 0 5.36 0 12S5.36 24 12 24 24 18.64 24 12 18.64 0 12 0M12 1.41C13.2 1.41 14.36 1.63 15.43 2L12.13 9.13C12.09 9.17 12.09 9.26 12 9.26S11.91 9.17 11.87 9.13L8.57 2C9.64 1.63 10.8 1.42 12 1.42M6.9 2.74L10.72 10.97C10.8 11.14 10.89 11.19 11 11.19H13C13.12 11.19 13.2 11.14 13.29 10.97L17.06 2.74C18.64 3.64 20 4.93 20.96 6.47L15.6 16.84C15.56 16.93 15.5 16.97 15.47 16.97C15.39 16.97 15.39 16.89 15.34 16.84L13.29 12.3C13.2 12.13 13.12 12.09 13 12.09H11C10.89 12.09 10.8 12.13 10.71 12.3L8.66 16.84C8.61 16.89 8.62 16.97 8.53 16.97C8.44 16.97 8.44 16.89 8.4 16.84L3 6.47C3.94 4.93 5.32 3.64 6.9 2.74M2.06 8.53L8.23 20.53C8.31 20.7 8.4 20.83 8.62 20.83C8.83 20.83 8.91 20.7 9 20.53L11.87 14.14C11.91 14.06 11.96 14 12 14C12.09 14 12.09 14.1 12.13 14.14L15.04 20.53C15.13 20.7 15.21 20.83 15.43 20.83C15.64 20.83 15.73 20.7 15.81 20.53L22 8.53C22.37 9.6 22.59 10.76 22.59 12C22.54 17.79 17.79 22.59 12 22.59C6.21 22.59 1.46 17.79 1.46 12C1.46 10.8 1.67 9.65 2.06 8.53Z'/></svg>"""
        ),
        'Mercedes': (
            """<svg viewBox='0 0 32 32' fill='none' xmlns='http://www.w3.org/2000/svg'><path fill-rule='evenodd' clip-rule='evenodd' d='M30 16C30 8.3 23.7 2 16 2C8.3 2 2 8.3 2 16C2 23.7 8.3 30 16 30C23.7 30 30 23.7 30 16ZM16 27.725C19.1063 27.725 22.0813 26.5 24.2688 24.3125C24.9775 23.6037 25.5803 22.8243 26.073 21.9911L15.9563 17.8375L5.90192 21.9655C6.39428 22.7988 6.99228 23.5735 7.6875 24.2687C9.91875 26.5 12.8938 27.725 16 27.725ZM5.62145 21.4629C4.74458 19.7961 4.275 17.9232 4.275 16C4.275 12.8937 5.5 9.91874 7.73125 7.77499C9.81269 5.69355 12.6071 4.48355 15.5492 4.3711L14.2063 14.8625L5.62145 21.4629ZM16.138 4.36327C19.193 4.39774 22.0706 5.57681 24.2688 7.77499C26.4563 9.96249 27.6813 12.9375 27.6813 16.0437C27.6813 17.9614 27.231 19.8124 26.3615 21.4732C24.8873 20.3034 17.9688 14.8188 17.9688 14.8188L16.138 4.36327Z' fill='#283544'/></svg>"""
        ),
    }

    def _get_brand_svg_xml(self, brand: str):
        if not brand:
            return None
        b_norm = str(brand).strip().capitalize()
        # próbuj też bez kapitalizacji
        for k, v in self.BRAND_SVG.items():
            if k.lower() == b_norm.lower():
                return v
        return None

    def _show_brand_overlay(self, brand):
        """Pokaż/aktualiz pół‑przezroczyste logo (inline SVG) – brak zależności od plików."""
        svg_xml = self._get_brand_svg_xml(brand)
        if not svg_xml:
            if getattr(self, 'brand_overlay', None):
                self.brand_overlay.hide()
            return
        from PyQt5.QtSvg import QSvgWidget
        from PyQt5.QtCore import QByteArray
        self._current_brand = brand
        if not getattr(self, 'brand_overlay', None):
            self.brand_overlay = QSvgWidget(self.image_label)
            self.brand_overlay.setStyleSheet("background: rgba(255,255,255,0.60); border: 1px solid rgba(22,34,46,0.08); border-radius: 20px;")
            self.brand_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.brand_overlay.setParent(self.image_label)
        try:
            self.brand_overlay.load(QByteArray(svg_xml.encode('utf-8')))
        except Exception:
            pass
        self.brand_overlay.show()
        self._reposition_brand_overlay()

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

        # dynamic theme will be applied after UI build
        self.theme = 'dark'
        # config
        self.config = self._load_config()
        self.device = self._choose_device(self.config.get('model', {}).get('device_preference', 'auto'))
        # services
        self.logger = setup_logger(
            level=self.config.get('logging', {}).get('level', 'INFO'),
            logfile=self.config.get('logging', {}).get('file', 'logs/app.log')
        )
        self.pred_service = PredictionService(
            yolo_weights=self.config.get('model', {}).get('yolo_weights', 'yolov8s.pt'),
            runs_dir=self.config.get('model', {}).get('runs_dir', 'runs'),
            device=self.device,
            logger=self.logger,
            min_conf=float(self.config.get('model', {}).get('min_conf', 0.25))
        )
        self.yolo = None  # legacy usage for existing PredictionThread
        self.classifier = None
        self.idx_to_label = None
        self.heatmap_visible = False
        self.feedback_saved = False
        self.brand_overlay = None
        self.log_viewer = None
        self._build_ui()
        # apply initial theme after widgets created
        self._apply_theme(self.theme)
        self._animations = []
        # ensure db exists
        try:
            init_db()
        except Exception:
            pass

        # Zablokuj dalsze zmiany rozmiaru – po zbudowaniu GUI ustawiamy stały rozmiar.
        # Używamy singleShot aby poczekać aż układ się policzy po show().
        QTimer.singleShot(0, self._lock_initial_size)

    def _lock_initial_size(self):
        try:
            self.setFixedSize(self.size())
        except Exception:
            pass
    # Usuwamy stałe blokowanie rozmiaru – okno ma być responsywne

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
        self.title_label.setStyleSheet("background: transparent;")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignHCenter)
        nav_layout.addWidget(self.title_label)

        self.model_info_label = QLabel("Model: (ładowanie...)")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("background: transparent;")
        nav_layout.addWidget(self.model_info_label)

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

        # Icon-only logs button
        self.logs_button = QToolButton()
        self.logs_button.setToolTip("Logi")
        try:
            self.logs_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        except Exception:
            pass
        try:
            self.logs_button.setIconSize(QSize(20, 20))
            self.logs_button.setAutoRaise(True)
        except Exception:
            pass
        self.logs_button.clicked.connect(self._open_logs_viewer)
        nav_layout.addWidget(self.logs_button)

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

        # Theme toggle as sun/moon icon-like button
        self.theme_button = QToolButton()
        self.theme_button.setToolTip("Motyw: Dark")
        try:
            self.theme_button.setAutoRaise(True)
        except Exception:
            pass
        self.theme_button.clicked.connect(self._toggle_theme)
        # set initial glyph matching current theme
        try:
            self._update_theme_button_icon()
        except Exception:
            # fallback text
            self.theme_button.setText('🌙')
        nav_layout.addWidget(self.theme_button)

        nav_layout.addStretch(1)

        # Content area
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(12)
        # Image frame & internals (responsywne – minimalny rozmiar tylko jako baza)
        self.image_frame = QFrame()
        self.image_frame.setObjectName("ImageFrame")
        self.image_frame.setMinimumSize(*self.IMAGE_AREA_MIN)
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

        from PyQt5.QtWidgets import QVBoxLayout as _QVBox
        img_layout = _QVBox(self.image_frame)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(0)
        self.image_label = QLabel()
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: transparent;")
        img_layout.addWidget(self.image_label)

        self.heatmap_label = QLabel(self.image_label)
        self.heatmap_label.setObjectName("HeatmapLabel")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.heatmap_label.hide()

        content_layout.addWidget(self.image_frame, alignment=Qt.AlignHCenter)

        self.result_label = QLabel("Gotowy – wybierz obraz")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        try:
            from PyQt5.QtWidgets import QSizePolicy as _QSP
            self.result_label.setSizePolicy(_QSP.Expanding, _QSP.Fixed)
            self.result_label.setMinimumHeight(90)
            self.result_label.setMaximumHeight(self.RESULT_MAX_HEIGHT)
            self.result_label.setWordWrap(True)
        except Exception:
            pass
        content_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        shell_layout.addWidget(nav)
        shell_layout.addLayout(content_layout, stretch=1)

        outer = QVBoxLayout()
        outer.setContentsMargins(30, 30, 30, 30)
        outer.addWidget(shell)
        self.setLayout(outer)

    # ---------------- THEME SYSTEM -----------------
    def _toggle_theme(self):
        self.theme = 'light' if self.theme == 'dark' else 'dark'
        self._apply_theme(self.theme)
        try:
            self._update_theme_button_icon()
        except Exception:
            pass

    def _update_theme_button_icon(self):
        """Update theme toggle button to show glyph matching the current theme.
        Dark -> moon, Light -> sun. Keep tooltip descriptive.
        """
        if not hasattr(self, 'theme_button'):
            return
        if self.theme == 'dark':
            # show moon for dark theme
            self.theme_button.setText('🌙')
            self.theme_button.setToolTip('Motyw: Dark (kliknij, aby przełączyć)')
        else:
            # show sun for light theme
            self.theme_button.setText('☀')
            self.theme_button.setToolTip('Motyw: Light (kliknij, aby przełączyć)')

    def _apply_theme(self, mode: str):
        """Apply modern, subtle glass theme. mode in {'dark','light'}."""
        dark = (mode == 'dark')
        # Palette tokens
        if dark:
            bg_gradient = "qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e1115, stop:1 #1b2733)"
            shell_bg = "rgba(255,255,255,0.06)"
            panel_bg = "rgba(255,255,255,0.04)"
            border_col = "rgba(255,255,255,0.10)"
            text_col = "#edf2f7"
            accent = "#0ea5e9"
            accent_hover = "#0d8fd0"
            accent_down = "#0b78b2"
            subtle = "#334155"
            result_bg = "rgba(255,255,255,0.85)"
            result_fg = "#0f172a"
            table_bg = "#1e293b"
            header_bg = "#2a3a4c"
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
            result_bg = "rgba(255,255,255,0.80)"
            result_fg = "#0f172a"
            table_bg = "#f1f5f9"
            header_bg = "#e2e8f0"

        radius_l = 28
        radius_m = 22
        radius_img = 26
        radius_inner = 18

        style = f"""
            QWidget {{
                background: {bg_gradient};
                font-family: 'Segoe UI','Arial';
                font-size: 14px;
                color: {text_col};
            }}
            QFrame#ShellFrame {{
                background: {shell_bg};
                border: 1px solid {border_col};
                border-radius: {radius_l}px;
                /* backdrop-filter removed (unsupported in Qt) */
            }}
            QFrame#NavPanel {{
                background: {panel_bg};
                border-right: 1px solid {border_col};
                border-radius: {radius_m}px;
            }}
            QLabel#TitleLabel {{
                font-size: 30px; font-weight: 600; letter-spacing: 0.5px; padding: 4px 4px 12px 4px;
            }}
            QFrame#ImageFrame {{
                background: {panel_bg};
                border: 1px solid {border_col};
                border-radius: {radius_img}px;
            }}
            QLabel#ImageLabel {{
                background: {'#0f172a' if dark else '#ffffff'};
                border-radius: {radius_inner}px;
                border: 1px solid {border_col};
            }}
            QLabel#HeatmapLabel {{
                border-radius: {radius_inner}px;
            }}
            QLabel#ResultLabel {{
                font-size: 15px; font-weight: 600;
                color: {result_fg};
                background: {result_bg};
                border: 1px solid {border_col};
                border-radius: {radius_inner}px; padding: 10px 18px; margin-top: 14px;
            }}
            QPushButton {{
                background: {accent};
                color: {'#f1f5f9' if dark else '#ffffff'};
                border: 0px solid transparent;
                border-radius: 14px;
                padding: 8px 16px;
                font-size: 13px; font-weight: 600;
                margin: 6px 4px;
                min-width: 140px;
            }}
            QPushButton:hover {{ background: {accent_hover}; }}
            QPushButton:pressed {{ background: {accent_down}; }}
            QPushButton:disabled {{ background: {subtle}; color: {'#64748b' if dark else '#94a3b8'}; }}
            QPushButton#DangerBtn {{ background: #dc2626; }}
            QPushButton#DangerBtn:hover {{ background: #b91c1c; }}
            QPushButton#SecondaryBtn {{ background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.04));
                color: {text_col};
            }}
            QPushButton#SecondaryBtn:hover {{ background: linear-gradient(135deg, rgba(255,255,255,0.22), rgba(255,255,255,0.08)); }}
            QPushButton#SecondaryBtn:pressed {{ background: linear-gradient(135deg, rgba(255,255,255,0.28), rgba(255,255,255,0.12)); }}
            QTableWidget {{
                background: {table_bg};
                gridline-color: {border_col};
                selection-background-color: {accent_hover};
                selection-color: #ffffff;
                border: 1px solid {border_col};
                border-radius: 12px;
            }}
            QHeaderView::section {{ background: {header_bg}; color: {text_col}; border: none; padding: 6px 8px; font-weight:600; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; margin:4px; }}
            QScrollBar::handle:vertical {{ background: {accent}; border-radius:5px; min-height:40px; }}
            QScrollBar::handle:vertical:hover {{ background: {accent_hover}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ background: none; height: 0; }}
            QMessageBox {{ background: {panel_bg}; }}
        """
        try:
            self.setStyleSheet(style)
        except Exception:
            pass

    # ------------------------------------------------

    def _reposition_brand_overlay(self):
        """Center SVG overlay within image_label bounds (respects margins)."""
        if not getattr(self, 'brand_overlay', None):
            return
        try:
            frame_w = self.image_label.width()
            frame_h = self.image_label.height()
            if frame_w <= 0 or frame_h <= 0:
                return
            max_ratio = getattr(self, 'BRAND_LOGO_MAX_RATIO', 0.55)
            # Per‑brand override (jeśli niższy)
            try:
                brand = getattr(self, '_current_brand', None)
                if brand:
                    for k, v in self.BRAND_LOGO_SCALE_OVERRIDES.items():
                        if k.lower() == str(brand).lower():
                            if v < max_ratio:
                                max_ratio = v
                            break
            except Exception:
                pass
            # Pobierz aspect ratio z viewBox SVG
            aspect = 1.0
            try:
                renderer = self.brand_overlay.renderer()
                if renderer:
                    vb = renderer.viewBox()
                    vw, vh = vb.width(), vb.height()
                    if vw > 0 and vh > 0:
                        aspect = vw / vh
            except Exception:
                pass
            max_w = frame_w * max_ratio
            max_h = frame_h * max_ratio
            # Start od maksymalnej szerokości
            w = max_w
            h = w / aspect
            if h > max_h:  # ogranicz wysokością
                h = max_h
                w = h * aspect
            # Min rozmiar aby pozostało widoczne
            w = max(60, int(w))
            h = max(60, int(h))
            self.brand_overlay.setFixedSize(w, h)
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
            self._update_image_frame_bounds()
            # Ensure heatmap overlay always fills image_label
            self.heatmap_label.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
            # Nie zmieniamy rozmiaru etykiety – tylko odświeżamy skalowanie jeżeli jest obraz
            if hasattr(self, 'current_image_path') and self.current_image_path:
                self._refresh_base_image()
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
            self._set_canvas_scaled(pix)
        except Exception:
            pass

    def _set_canvas_scaled(self, pix: QPixmap):
        """Skaluj obraz do aktualnego rozmiaru etykiety zachowując proporcje (bez tworzenia dodatkowego 'canvas')."""
        if not pix or pix.isNull():
            return
        # Docelowy rozmiar – preferuj image_label, fallback do frame, fallback do minimum
        target_size = self.image_label.size()
        if target_size.width() < 10 or target_size.height() < 10:
            frame_size = self.image_frame.size()
            if frame_size.width() >= 10 and frame_size.height() >= 10:
                target_size = frame_size
            else:
                from PyQt5.QtCore import QSize
                target_size = QSize(*self.IMAGE_AREA_MIN)
        scaled = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setAlignment(Qt.AlignCenter)

    def _update_image_frame_bounds(self):
        """Dopasuj rozmiar ramki obrazu, aby nie rozciągała się do końca okna, utrzymując proporcje."""
        try:
            total_w = max(1, self.width())
            total_h = max(1, self.height())
            # Dostępna szerokość po odjęciu panelu nawigacji i marginesów shell
            nav_w = getattr(self, 'title_label', None).parent().width() if getattr(self, 'title_label', None) else int(total_w * 0.25)
            shell_margins = 18 * 2 + 30 * 2  # shell + outer
            avail_w = max(200, total_w - nav_w - shell_margins)
            max_w_cap = int(total_w * self.IMAGE_WIDTH_RATIO)
            # Docelowa powierzchnia (procent całego okna)
            desired_area = self.IMAGE_TARGET_AREA_RATIO * total_w * total_h
            # Wyprowadź wymiary z zachowaniem proporcji: area = w*h, h = w/aspect
            # => area = w^2 / aspect  => w = sqrt(area * aspect)
            import math
            target_w = math.sqrt(max(1.0, desired_area) * self.IMAGE_ASPECT)
            target_h = target_w / self.IMAGE_ASPECT
            # Ogranicz przez dostępne wymiary oraz maksymalne limity szer/wys
            max_h_cap = int(total_h * 0.80)
            if target_w > avail_w:
                target_w = avail_w
                target_h = target_w / self.IMAGE_ASPECT
            if target_w > max_w_cap:
                target_w = max_w_cap
                target_h = target_w / self.IMAGE_ASPECT
            if target_h > max_h_cap:
                target_h = max_h_cap
                target_w = target_h * self.IMAGE_ASPECT
            # Zachowaj minimalny sensowny wymiar
            min_w = self.IMAGE_AREA_MIN[0] // 2
            min_h = self.IMAGE_AREA_MIN[1] // 2
            target_w = max(min_w, int(target_w))
            target_h = max(min_h, int(target_h))
            # ustaw rozmiary
            self.image_frame.setMaximumSize(target_w, target_h)
            self.image_frame.setMinimumSize(min(self.IMAGE_AREA_MIN[0], target_w), min(self.IMAGE_AREA_MIN[1], target_h))
            self.image_frame.resize(target_w, target_h)
            self.image_label.resize(target_w, target_h)
        except Exception:
            pass

    def load_models(self):
        # keep legacy method for existing threads; prefer prediction_service
        if self.yolo is None:
            try:
                self.pred_service.load_yolo()
                self.yolo = self.pred_service._yolo
                self._update_model_info()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Nie udało się załadować YOLO: {e}")
                self.yolo = None

    def _update_model_info(self):
        try:
            if self.model_info_label:
                self.model_info_label.setText(f"Model: {self.pred_service.model_info}")
        except Exception:
            pass

    def _choose_device(self, pref: str):
        if pref == 'cpu':
            return 'cpu'
        if pref == 'cuda' or pref == 'gpu' or pref == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cpu'

    def _load_config(self):
        path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

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

        self.batch_thread = BatchPredictionThread(paths, self.pred_service)
        self.batch_thread.finished.connect(self._on_batch_finished)
        self.batch_thread.error.connect(self._on_batch_error)
        try:
            self.batch_thread.progress.connect(self._on_batch_progress)
        except Exception:
            pass
        self.batch_thread.start()

    def _set_image_and_predict(self, path: str):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            QMessageBox.critical(self, 'Błąd', f'Nie można otworzyć obrazu: {e}')
            return
        self.current_image = img
        self.current_image_path = path
        try:
            self._set_canvas_scaled(QPixmap(path))
        except Exception:
            pass
        # animate smooth fade-in for the newly loaded image
    # (Opcjonalnie) Można wyłączyć animacje – pomijamy fade
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
        self.pred_thread = PredictionThread(self.current_image_path, self.pred_service)
        self.pred_thread.finished.connect(self._on_pred_finished)
        self.pred_thread.error.connect(self._on_pred_error)
        self.pred_thread.start()

    def _open_test_mode(self):
        # Ensure a folder exists for test images (samples/)
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            samples_dir = os.path.join(root, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
        except Exception:
            pass
        # Modeless, live test dialog: clicking an image triggers prediction, dialog stays open
        try:
            # Jeśli już otwarty – pokaż istniejące okno
            if getattr(self, '_sample_dlg', None) is not None and self._sample_dlg.isVisible():
                self._sample_dlg.raise_()
                self._sample_dlg.activateWindow()
                return
        except Exception:
            pass
        try:
            self._sample_dlg = SampleDialog(self, live_mode=True)
            self._sample_dlg.imageSelected.connect(self._set_image_and_predict)
            # NOWE: obsługa batch po multi-zaznaczeniu
            self._sample_dlg.imagesSelected.connect(self._run_batch)
            self._sample_dlg.setModal(False)
            self._sample_dlg.show()
        except Exception as e:
            QMessageBox.warning(self, 'Tryb testowy', f'Nie można otworzyć okna testowego: {e}')

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

    def _on_batch_progress(self, current: int, total: int):
        try:
            self.result_label.setText(f'Batch: {current}/{total}')
        except Exception:
            pass

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
        auto_mark = bool(self.config.get('app', {}).get('auto_mark_correct', True))
        if auto_mark and not self.feedback_saved:
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
            try:
                dst = save_image_copy(self.current_image_path)
            except Exception:
                base = os.path.basename(self.current_image_path)
                dst = save_image_copy(self.current_image_path, prefix='copied')
            rid = insert_record(self.current_image_path, dst, brand, float(conf or 0.0), True, float(proc_time), int(time.time()))
            try:
                self.last_result['record_id'] = int(rid)
            except Exception:
                pass
        except Exception:
            pass
        if brand is None:
            self.result_label.setText(res.get('message', 'Brak wyników'))
        else:
            try:
                self.result_label.setText(f"Marka:{brand}, Pewność:{float(conf or 0.0):.2f}%")
            except Exception:
                self.result_label.setText(f"Marka:{brand}")

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
    # Pomijamy animację dla uproszczenia
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

    def _open_logs_viewer(self):
        """Modeless viewer logów na żywo (tail)."""
        try:
            path = self.config.get('logging', {}).get('file', 'logs/app.log') if getattr(self, 'config', None) else 'logs/app.log'
            if self.log_viewer and self.log_viewer.isVisible():
                self.log_viewer.raise_()
                self.log_viewer.activateWindow()
                return
            self.log_viewer = LiveLogViewer(path, self)
            self.log_viewer.show()
        except Exception as e:
            QMessageBox.warning(self, 'Logi', f'Błąd: {e}')

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

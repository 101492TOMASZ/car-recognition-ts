from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from predict import predict_image, find_latest_best_checkpoint, load_classifier


class PredictionService:
    def __init__(self, yolo_weights: str, runs_dir: str, device: str = "cpu", logger=None, min_conf: float = 0.25):
        self._yolo = None
        self._classifier = None
        self._idx_to_label = None
        self.yolo_weights = yolo_weights
        self.runs_dir = runs_dir
        self.device = device
        self.logger = logger
        self.min_conf = min_conf

    def load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.yolo_weights)
            try:
                self._yolo.to(self.device)
            except Exception:
                pass
            if self.logger:
                self.logger.info(f"YOLO loaded: {self.yolo_weights} on {self.device}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load YOLO: {e}")
            self._yolo = None

    def load_classifier(self):
        best, label_map = find_latest_best_checkpoint(self.runs_dir)
        if best is None:
            if self.logger:
                self.logger.warning("No classifier checkpoint found.")
            return
        if self._classifier is not None:
            return
        try:
            self._classifier, self._idx_to_label = load_classifier(best, label_map, device=self.device)
            if self.logger:
                self.logger.info(f"Classifier loaded: {best} (labels: {len(self._idx_to_label)})")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load classifier: {e}")

    def ensure_models(self):
        self.load_yolo()
        self.load_classifier()

    def predict(self, image_path: str) -> Dict[str, Any]:
        self.ensure_models()
        return predict_image(
            image_path,
            self._yolo,
            self._classifier,
            self._idx_to_label,
            device=self.device,
            min_conf=self.min_conf,
            verbose=True if self.logger else False,
            logger=self.logger,
        )

    @property
    def model_info(self) -> str:
        parts = []
        parts.append(f"Urządzenie={self.device}")
        parts.append(f"STATUS YOLO={'ok' if self._yolo else 'none'}")
        parts.append(f"clf={'ok' if self._classifier else 'none'}")
        return ", ".join(parts)
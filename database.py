import sqlite3
from datetime import datetime
import os
from PIL import Image
import io

class Database:
    def __init__(self, db_file="predictions.db"):
        self.db_file = db_file
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    image_path TEXT,
                    image_data BLOB,
                    brand TEXT,
                    confidence REAL
                )
            ''')
            conn.commit()

    def save_prediction(self, image_path, brand, confidence):
        # Read and resize image for storage
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img.thumbnail((300, 300))  # Resize for storage
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, image_path, image_data, brand, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), image_path, img_byte_arr, brand, confidence))
            conn.commit()

    def get_all_predictions(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
            return cursor.fetchall()
import sqlite3
from datetime import datetime
import os
import appdirs

class Database:
    def __init__(self):
        # Utworzenie folderu dla aplikacji w AppData
        app_name = "CarRecognition"
        app_author = "TomaszSlocinski"
        data_dir = appdirs.user_data_dir(app_name, app_author)
        
        # Utworzenie folderu jeśli nie istnieje
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Ścieżka do pliku bazy danych
        self.db_file = os.path.join(data_dir, "predictions.db")
        self.init_database()

    def init_database(self):
        """Inicjalizacja bazy danych i utworzenie tabeli jeśli nie istnieje"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    brand TEXT,
                    confidence REAL,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def save_prediction(self, image_path, brand, confidence):
        """Zapisuje wynik predykcji do bazy danych"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (image_path, brand, confidence, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (image_path, brand, confidence, datetime.now()))
            conn.commit()

    def get_all_predictions(self):
        """Pobiera wszystkie predykcje z bazy danych"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
            return cursor.fetchall()
import sqlite3
from datetime import datetime
import os
from PIL import Image
import io
from car_detector import CarDetector

class Database:
    def __init__(self, db_file="predictions.db"):
        # Set database file path to be in the temp directory relative to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = os.path.join(script_dir, "temp")
        self.img_dir = os.path.join(self.temp_dir, "img")
        self.ensure_temp_dirs()  # Create temp directories if they don't exist
        self.db_file = os.path.join(self.temp_dir, db_file)
        self.car_detector = CarDetector()
        self.init_database()

    def ensure_temp_dirs(self):
        """Create temporary directories if they don't exist"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

    def save_temp_image(self, image_data, prediction_id):
        """Save image data to a temporary file for display/export"""
        temp_path = os.path.join(self.img_dir, f"prediction_{prediction_id}.jpg")
        try:
            if isinstance(image_data, str) and os.path.exists(image_data):
                # If image_data is a path, copy the file
                import shutil
                shutil.copy2(image_data, temp_path)
            elif image_data is not None:
                # If we have bytes, write them directly
                if isinstance(image_data, bytes):
                    with open(temp_path, 'wb') as f:
                        f.write(image_data)
                else:
                    print(f"Invalid image data type: {type(image_data)}")
                    return None
            return temp_path
        except Exception as e:
            print(f"Error saving temp image: {e}")
            return None

    def get_image_path(self, prediction_id, image_data):
        """Get or create temporary image file path"""
        temp_path = os.path.join(self.img_dir, f"prediction_{prediction_id}.jpg")
        if not os.path.exists(temp_path):
            return self.save_temp_image(image_data, prediction_id)
        return temp_path

    def save_prediction(self, image_path, brand, confidence):
        # First check if image contains a car
        has_car, det_conf = self.car_detector.detect_car(image_path)
        
        if not has_car:
            raise ValueError("Brak auta na zdjęciu!")

        # Read and process the image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img.thumbnail((300, 300))  # Resize for storage
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, image_data, brand, confidence, is_bad_prediction)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), img_byte_arr, brand, confidence, False))
            conn.commit()
            new_id = cursor.lastrowid
            
            # Save image file right away
            self.save_temp_image(img_byte_arr, new_id)
            return new_id

    def get_all_predictions(self, descending=True):
        order = 'DESC' if descending else 'ASC'
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            predictions = cursor.execute(f'SELECT * FROM predictions ORDER BY timestamp {order}').fetchall()
            # Create or get image files for each prediction
            result = []
            for pred in predictions:
                pred_id = pred[0]
                timestamp = pred[1]
                image_data = pred[2]
                brand = pred[3]
                confidence = pred[4]
                is_bad = pred[5]
                
                # Get image path
                temp_path = self.get_image_path(pred_id, image_data)
                
                # Ensure confidence is a float
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                    
                # Create a new tuple with properly formatted data
                result.append((
                    pred_id,
                    timestamp,
                    temp_path,
                    brand,
                    confidence,
                    bool(is_bad)
                ))
            return result

    def init_database(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            # Remove image_path column as we'll store and handle images directly
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    image_data BLOB,
                    brand TEXT,
                    confidence REAL,
                    is_bad_prediction BOOLEAN DEFAULT 0
                )
            ''')
            conn.commit()

    def mark_prediction_as_bad(self, prediction_id, is_bad=True):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions 
                SET is_bad_prediction = ?
                WHERE id = ?
            ''', (is_bad, prediction_id))
            conn.commit()
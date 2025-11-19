"""
FarmAI Analytics Platform - Database Manager
Complete 8-table relational database with CRUD operations
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, List


# ----------------- Logging setup (ensure logs dir exists) -----------------
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "app.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FarmAIDatabaseManager:
    """
    Complete database manager for FarmAI Analytics Platform
    Manages 8 tables with full CRUD operations
    """

    def __init__(self, db_path: str = 'farmer_analytics.db'):
        # store as Path for consistent behavior
        self.db_path = Path(db_path)
        # ensure parent directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # initialize database (creates file and tables if not present)
        self.init_database()
        logger.info("Database manager initialized: %s", str(self.db_path))

    def _get_conn(self):
        """Return a new sqlite connection (use context manager where possible)."""
        return sqlite3.connect(str(self.db_path))

    def init_database(self):
        """Initialize database with all 8 tables"""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # TABLE 1: Farmers
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS farmers (
                    farmer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT,
                    village TEXT,
                    district TEXT,
                    state TEXT,
                    crops_grown TEXT,
                    farm_size_acres REAL,
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME
                )
            ''')

            # TABLE 2: Crops
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crops (
                    crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crop_name TEXT UNIQUE NOT NULL,
                    season TEXT,
                    growth_period_days INTEGER,
                    common_diseases TEXT,
                    optimal_temperature REAL,
                    optimal_rainfall REAL
                )
            ''')

            # TABLE 3: Diseases
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diseases (
                    disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_name TEXT UNIQUE NOT NULL,
                    affected_crops TEXT,
                    symptoms TEXT,
                    causes TEXT,
                    treatment_options TEXT,
                    prevention_methods TEXT,
                    severity_level TEXT,
                    image_characteristics TEXT
                )
            ''')

            # TABLE 4: Farmer Queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS farmer_queries (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmer_id TEXT NOT NULL,
                    crop_id INTEGER,
                    disease_id INTEGER,
                    query_text TEXT,
                    query_language TEXT,
                    response_text TEXT,
                    response_time_seconds REAL,
                    confidence_score FLOAT,
                    helpful BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id),
                    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
                    FOREIGN KEY (disease_id) REFERENCES diseases(disease_id)
                )
            ''')

            # TABLE 5: Disease Predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disease_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmer_id TEXT NOT NULL,
                    crop_id INTEGER,
                    image_filename TEXT,
                    predicted_disease_id INTEGER,
                    confidence_score FLOAT,
                    model_version TEXT,
                    prediction_time_seconds REAL,
                    actual_disease_id INTEGER,
                    is_correct BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id),
                    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
                    FOREIGN KEY (predicted_disease_id) REFERENCES diseases(disease_id),
                    FOREIGN KEY (actual_disease_id) REFERENCES diseases(disease_id)
                )
            ''')

            # TABLE 6: Treatment Recommendations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS treatment_recommendations (
                    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_id INTEGER NOT NULL,
                    treatment_name TEXT,
                    treatment_steps TEXT,
                    cost_estimate REAL,
                    effectiveness_percentage FLOAT,
                    time_to_recovery_days INTEGER,
                    recommended_product TEXT,
                    region_applicable TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (disease_id) REFERENCES diseases(disease_id)
                )
            ''')

            # TABLE 7: Chatbot Logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chatbot_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    farmer_id TEXT,
                    query_text TEXT,
                    response_text TEXT,
                    query_language TEXT,
                    response_quality_score FLOAT,
                    response_time_seconds REAL,
                    model_used TEXT,
                    tokens_used INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id)
                )
            ''')

            # TABLE 8: Analytics Metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_date DATE,
                    total_queries INTEGER,
                    total_farmers INTEGER,
                    average_response_time REAL,
                    model_accuracy REAL,
                    top_disease TEXT,
                    total_predictions INTEGER,
                    successful_predictions INTEGER,
                    feature_used TEXT,
                    metric_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
        logger.info("All 8 tables created successfully")

    # ==================== FARMERS TABLE OPERATIONS ====================

    def add_farmer(self, farmer_id: str, name: str, phone: str = None,
                   village: str = None, district: str = None, state: str = None,
                   crops: str = None, farm_size: float = None) -> bool:
        """Add new farmer to database"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO farmers 
                    (farmer_id, name, phone, village, district, state, crops_grown, farm_size_acres, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (farmer_id, name, phone, village, district, state, crops, farm_size, datetime.now()))
                conn.commit()
            logger.info("Farmer added: %s", farmer_id)
            return True
        except sqlite3.IntegrityError:
            # Farmer already exists, update last_active
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE farmers SET last_active = ? WHERE farmer_id = ?
                    ''', (datetime.now(), farmer_id))
                    conn.commit()
                logger.info("Farmer updated last_active: %s", farmer_id)
                return True
            except Exception as e:
                logger.error("Error updating farmer last_active: %s", str(e))
                return False
        except Exception as e:
            logger.error("Error adding farmer: %s", str(e))
            return False

    def get_farmer(self, farmer_id: str) -> Optional[Dict]:
        """Get farmer information"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM farmers WHERE farmer_id = ?', (farmer_id,))
                result = cursor.fetchone()
            if result:
                return {
                    'farmer_id': result[0],
                    'name': result[1],
                    'phone': result[2],
                    'village': result[3],
                    'district': result[4],
                    'state': result[5],
                    'crops_grown': result[6],
                    'farm_size_acres': result[7],
                    'created_date': result[8],
                    'last_active': result[9]
                }
        except Exception as e:
            logger.error("Error getting farmer %s: %s", farmer_id, str(e))
        return None

    # ==================== DISEASES TABLE OPERATIONS ====================

    def add_disease(self, disease_name: str, affected_crops: str = None,
                    symptoms: str = None, causes: str = None,
                    treatment: str = None, prevention: str = None,
                    severity: str = 'Medium', image_chars: str = None) -> Optional[int]:
        """Add disease to knowledge base"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO diseases 
                    (disease_name, affected_crops, symptoms, causes, treatment_options, 
                     prevention_methods, severity_level, image_characteristics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (disease_name, affected_crops, symptoms, causes, treatment,
                      prevention, severity, image_chars))
                conn.commit()
                disease_id = cursor.lastrowid
            logger.info("Disease added: %s", disease_name)
            return disease_id
        except sqlite3.IntegrityError:
            # Disease exists, get its ID
            try:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT disease_id FROM diseases WHERE disease_name = ?', (disease_name,))
                    result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                logger.error("Error fetching existing disease id: %s", str(e))
                return None
        except Exception as e:
            logger.error("Error adding disease: %s", str(e))
            return None

    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get detailed disease information"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM diseases WHERE disease_name = ?', (disease_name,))
                result = cursor.fetchone()
            if result:
                return {
                    'disease_id': result[0],
                    'disease_name': result[1],
                    'affected_crops': result[2],
                    'symptoms': result[3],
                    'causes': result[4],
                    'treatment_options': result[5],
                    'prevention_methods': result[6],
                    'severity_level': result[7],
                    'image_characteristics': result[8]
                }
        except Exception as e:
            logger.error("Error getting disease info %s: %s", disease_name, str(e))
        return None

    # ==================== FARMER QUERIES TABLE OPERATIONS ====================

    def save_query(self, farmer_id: str, query_text: str, response_text: str,
                   response_time: float, confidence: float = 0.85,
                   crop_id: int = None, disease_id: int = None,
                   language: str = 'English', helpful: bool = None) -> bool:
        """Save farmer query and chatbot response"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO farmer_queries
                    (farmer_id, crop_id, disease_id, query_text, query_language, 
                     response_text, response_time_seconds, confidence_score, helpful)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (farmer_id, crop_id, disease_id, query_text, language,
                      response_text, response_time, confidence, helpful))
                conn.commit()
            logger.info("Query saved for farmer %s", farmer_id)
            return True
        except Exception as e:
            logger.error("Error saving query: %s", str(e))
            return False

    # ==================== DISEASE PREDICTIONS TABLE OPERATIONS ====================

    def save_prediction(self, farmer_id: str, predicted_disease_id: Optional[int],
                       confidence: float, model_version: str,
                       prediction_time: float, image_file: str = None,
                       crop_id: int = None, actual_disease_id: Optional[int] = None,
                       is_correct: Optional[bool] = None) -> bool:
        """Save disease prediction"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO disease_predictions
                    (farmer_id, crop_id, image_filename, predicted_disease_id, 
                     confidence_score, model_version, prediction_time_seconds,
                     actual_disease_id, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (farmer_id, crop_id, image_file, predicted_disease_id,
                      confidence, model_version, prediction_time,
                      actual_disease_id, is_correct))
                conn.commit()
            logger.info("Prediction saved for farmer %s", farmer_id)
            return True
        except Exception as e:
            logger.error("Error saving prediction: %s", str(e))
            return False

    # ==================== CHATBOT LOGS TABLE OPERATIONS ====================

    def log_chatbot_interaction(self, farmer_id: str, query: str, response: str,
                                language: str, response_time: float,
                                model_used: str = 'gemini-pro',
                                tokens: int = 0, quality_score: Optional[float] = None) -> bool:
        """Log chatbot interaction"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chatbot_logs
                    (farmer_id, query_text, response_text, query_language,
                     response_quality_score, response_time_seconds, model_used, tokens_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (farmer_id, query, response, language, quality_score,
                      response_time, model_used, tokens))
                conn.commit()
            logger.info("Chatbot interaction logged for farmer %s", farmer_id)
            return True
        except Exception as e:
            logger.error("Error logging chatbot: %s", str(e))
            return False

    # ==================== ANALYTICS OPERATIONS ====================

    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()

                # Total queries
                cursor.execute('SELECT COUNT(*) FROM farmer_queries')
                total_queries = cursor.fetchone()[0]

                # Total farmers
                cursor.execute('SELECT COUNT(*) FROM farmers')
                total_farmers = cursor.fetchone()[0]

                # Average response time
                cursor.execute('SELECT AVG(response_time_seconds) FROM farmer_queries WHERE response_time_seconds IS NOT NULL')
                avg_response_time = cursor.fetchone()[0] or 0

                # Total predictions
                cursor.execute('SELECT COUNT(*) FROM disease_predictions')
                total_predictions = cursor.fetchone()[0]

                # Model accuracy
                cursor.execute('''
                    SELECT COUNT(*) FROM disease_predictions 
                    WHERE is_correct = 1 AND is_correct IS NOT NULL
                ''')
                correct_predictions = cursor.fetchone()[0]

                accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

                # Top disease
                cursor.execute('''
                    SELECT d.disease_name, COUNT(*) as count 
                    FROM disease_predictions dp
                    LEFT JOIN diseases d ON dp.predicted_disease_id = d.disease_id
                    WHERE d.disease_name IS NOT NULL
                    GROUP BY d.disease_name 
                    ORDER BY count DESC 
                    LIMIT 1
                ''')
                top_disease_result = cursor.fetchone()
                top_disease = top_disease_result[0] if top_disease_result else 'N/A'

                # Average confidence
                cursor.execute('SELECT AVG(confidence_score) FROM disease_predictions')
                avg_confidence = cursor.fetchone()[0] or 0

            return {
                'total_queries': total_queries,
                'total_farmers': total_farmers,
                'avg_response_time': round(avg_response_time, 2),
                'total_predictions': total_predictions,
                'model_accuracy': round(accuracy, 2),
                'top_disease': top_disease,
                'avg_confidence': round(avg_confidence, 4)
            }
        except Exception as e:
            logger.error("Error computing analytics summary: %s", str(e))
            return {
                'total_queries': 0,
                'total_farmers': 0,
                'avg_response_time': 0,
                'total_predictions': 0,
                'model_accuracy': 0,
                'top_disease': 'N/A',
                'avg_confidence': 0
            }

    def get_time_series_data(self, days: int = 30) -> pd.DataFrame:
        """Get time series data for trends"""
        with self._get_conn() as conn:
            query = '''
                SELECT 
                    DATE(timestamp) as date, 
                    COUNT(*) as queries,
                    AVG(response_time_seconds) as avg_time,
                    AVG(confidence_score) as avg_confidence
                FROM farmer_queries
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(days,))
        return df

    def get_disease_distribution(self, limit: int = 10) -> pd.DataFrame:
        """Get disease distribution for analytics"""
        with self._get_conn() as conn:
            query = '''
                SELECT 
                    d.disease_name,
                    COUNT(*) as count,
                    AVG(dp.confidence_score) as avg_confidence,
                    d.severity_level
                FROM disease_predictions dp
                LEFT JOIN diseases d ON dp.predicted_disease_id = d.disease_id
                WHERE d.disease_name IS NOT NULL
                GROUP BY d.disease_name
                ORDER BY count DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df

    def get_farmer_history(self, farmer_id: str, limit: int = 50) -> pd.DataFrame:
        """Get complete history for a farmer"""
        with self._get_conn() as conn:
            query = '''
                SELECT 
                    query_id, query_text, response_text, 
                    query_language, response_time_seconds,
                    confidence_score, timestamp
                FROM farmer_queries
                WHERE farmer_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(farmer_id, limit))
        return df

    def export_analytics_data(self, filepath: str = 'dashboards/dashboard_data.csv') -> str:
        """Export complete analytics data for Power BI/Tableau"""
        with self._get_conn() as conn:
            query = '''
                SELECT 
                    fq.query_id, fq.farmer_id, fq.query_text, fq.response_text,
                    fq.query_language, fq.response_time_seconds, fq.confidence_score,
                    fq.timestamp, d.disease_name, d.severity_level,
                    c.crop_name, f.village, f.district, f.state
                FROM farmer_queries fq
                LEFT JOIN diseases d ON fq.disease_id = d.disease_id
                LEFT JOIN crops c ON fq.crop_id = c.crop_id
                LEFT JOIN farmers f ON fq.farmer_id = f.farmer_id
                ORDER BY fq.timestamp DESC
            '''
            df = pd.read_sql_query(query, conn)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info("Analytics data exported to %s", filepath)
        return filepath

    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create database backup"""
        if backup_path is None:
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            import shutil
            shutil.copy2(str(self.db_path), str(backup_path))
            logger.info("Database backed up to %s", str(backup_path))
            return True
        except Exception as e:
            logger.error("Backup failed: %s", str(e))
            return False




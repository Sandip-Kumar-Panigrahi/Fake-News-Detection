import os
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "predictions.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    try:
        cursor.execute("ALTER TABLE prediction_logs ADD COLUMN country TEXT DEFAULT 'Global'")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def save_prediction(news_text: str, prediction: str, confidence: float, country: str = "Global"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prediction_logs (news_text, prediction, confidence, created_at, country)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            news_text,
            prediction,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            country or "Global",
        ),
    )
    conn.commit()
    conn.close()


def create_user(username: str, email: str, password: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    already_exists = cursor.fetchone()
    if already_exists:
        conn.close()
        return False, "Email already registered."

    hashed_password = generate_password_hash(password)
    cursor.execute(
        """
        INSERT INTO users (username, email, password, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (username, email, hashed_password, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    return True, "Signup successful."


def authenticate_user(email: str, password: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return None

    if not check_password_hash(user_row["password"], password):
        return None

    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"],
    }

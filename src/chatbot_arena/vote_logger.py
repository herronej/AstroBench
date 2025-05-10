import sqlite3
from datetime import datetime
import os
import shutil
import threading
import time

DB_PATH = "votes.db"
BACKUP_DIR = "backups"
BACKUP_INTERVAL_SECONDS = 4 * 60 * 60  # 2 hours

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            question_type TEXT,
            question TEXT,
            correct_answer TEXT,
            explanation TEXT,
            model_a TEXT,
            model_a_response TEXT,
            model_b TEXT,
            model_b_response TEXT,
            vote TEXT
        );
        """)

def log_vote(session_id, question_type, question, correct_answer, explanation,
             model_a, model_a_response, model_b, model_b_response, vote):
    timestamp = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
        INSERT INTO votes (
            session_id, timestamp, question_type, question,
            correct_answer, explanation, model_a, model_a_response,
            model_b, model_b_response, vote
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, timestamp, question_type, question, correct_answer, explanation,
            model_a, model_a_response, model_b, model_b_response, vote
        ))
        conn.commit()

def get_vote_count():
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute("SELECT COUNT(*) FROM votes").fetchone()
        return result[0] if result else 0

def get_recent_votes(limit=10):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT * FROM votes ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()

def backup_votes_db():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    while True:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"votes_backup_{timestamp}.db")
        try:
            shutil.copyfile(DB_PATH, backup_file)
            print(f"[Backup] Created backup: {backup_file}")
        except Exception as e:
            print(f"[Backup] Failed to create backup: {e}")
        time.sleep(BACKUP_INTERVAL_SECONDS)

# Start backup thread if imported
init_db()
backup_thread = threading.Thread(target=backup_votes_db, daemon=True)
backup_thread.start()

if __name__ == "__main__":
    print(f"Total votes: {get_vote_count()}")
    for row in get_recent_votes():
        print(row)

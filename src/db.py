import aiosqlite
import logging

logger = logging.getLogger(__name__)

DB_PATH = "vault.db"
_db_conn = None

async def init_db(db_path: str = DB_PATH):
    """Initializes the database schema, WAL mode, and busy timeout."""
    global _db_conn
    if _db_conn is not None:
        await _db_conn.close()
    _db_conn = await aiosqlite.connect(db_path)
    # Pragma setup for concurrency
    await _db_conn.execute("PRAGMA journal_mode=WAL;")
    await _db_conn.execute("PRAGMA busy_timeout=10000;")
        
    # Schemas
    await _db_conn.executescript("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            namespace TEXT NOT NULL,
            filepath TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skill_usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_logs_session_timestamp ON skill_usage_logs(session_id, timestamp);

        CREATE TABLE IF NOT EXISTS hotbar_cache (
            namespace TEXT PRIMARY KEY,
            skills_list TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS session_metadata (
            session_id TEXT PRIMARY KEY,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_metadata_last_active ON session_metadata(last_active);
    """)
    await _db_conn.commit()
    logger.info(f"Database initialized successfully at {db_path}")
    return _db_conn

def get_db():
    return _db_conn

async def close_db():
    global _db_conn
    if _db_conn:
        await _db_conn.close()
        _db_conn = None

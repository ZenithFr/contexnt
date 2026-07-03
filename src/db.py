import aiosqlite
import logging

logger = logging.getLogger(__name__)

DB_PATH = "vault.db"

async def init_db(db_path: str = DB_PATH):
    """Initializes the database schema, WAL mode, and busy timeout."""
    async with aiosqlite.connect(db_path) as db:
        # Pragma setup for concurrency
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=10000;")
        
        # Schemas
        await db.executescript("""
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
        await db.commit()
        logger.info(f"Database initialized successfully at {db_path}")

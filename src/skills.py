import aiosqlite
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

async def index_skills_on_startup(db_path: str, vault_path: Path):
    """
    Two-way sync:
    1. Scans vault_path for SKILL.md files and upserts into `skills` table.
    2. Deletes records from `skills` table if the filepath no longer exists.
    """
    logger.info("Starting two-way skill sync...")
    
    found_skills = []
    async with aiosqlite.connect(db_path) as db:
        # 1. Upsert files found on disk
        if vault_path.exists():
            for md_file in vault_path.rglob("SKILL.md"):
                namespace = md_file.parent.name
                name = md_file.parent.name
                filepath = str(md_file.resolve())
                found_skills.append(filepath)
                
                await db.execute("""
                    INSERT INTO skills (name, namespace, filepath)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        namespace=excluded.namespace,
                        filepath=excluded.filepath
                """, (name, namespace, filepath))
            await db.commit()
                
        # 2. Delete stale records
        async with db.execute("SELECT id, filepath FROM skills") as cursor:
            rows = await cursor.fetchall()
            
        for row in rows:
            skill_id, filepath = row
            if filepath not in found_skills:
                # File deleted on disk
                await db.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
                logger.info(f"Removed stale skill from DB: {filepath}")
        await db.commit()
        
    logger.info("Two-way skill sync completed.")

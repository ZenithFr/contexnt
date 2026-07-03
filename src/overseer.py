import asyncio
import logging
import json
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)

async def overseer_loop(db_path: str, vault_path: Path, shutdown_event: asyncio.Event):
    """
    Background daemon for Continuous Improvement, Caching, and Cleanup.
    """
    logger.info("Overseer daemon started.")
    idle_threshold_seconds = 600

    while not shutdown_event.is_set():
        try:
            # 1. Clean up idle sessions
            async with aiosqlite.connect(db_path) as db:
                async with db.execute("""
                    SELECT session_id FROM session_metadata 
                    WHERE (strftime('%s', 'now') - strftime('%s', last_active)) > ?
                """, (idle_threshold_seconds,)) as cursor:
                    idle_sessions = await cursor.fetchall()
                
                for (session_id,) in idle_sessions:
                    target_file = vault_path / f"active_context_{session_id}.md"
                    if target_file.exists():
                        target_file.unlink()
                        logger.info(f"Overseer: Cleaned up idle session file {target_file}")
                    
                    await db.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
                
                await db.commit()

            # 2. Ranking & Bundling (Update Hotbar Cache)
            # Find namespaces where:
            # - Total unique skills used >= 5
            # - Used across >= 10 sessions
            # - Used across >= 5 distinct days
            async with aiosqlite.connect(db_path) as db:
                async with db.execute("SELECT DISTINCT namespace FROM skills") as cursor:
                    namespaces = await cursor.fetchall()
                    
                for (ns,) in namespaces:
                    query = """
                        SELECT 
                            COUNT(DISTINCT l.session_id) as session_count,
                            COUNT(DISTINCT date(l.timestamp)) as day_count,
                            COUNT(DISTINCT l.skill_name) as skill_count
                        FROM skill_usage_logs l
                        JOIN skills s ON l.skill_name = s.name
                        WHERE s.namespace = ?
                    """
                    async with db.execute(query, (ns,)) as cursor:
                        row = await cursor.fetchone()
                        
                    if row:
                        session_count, day_count, skill_count = row
                        if (session_count or 0) >= 10 and (day_count or 0) >= 5 and (skill_count or 0) >= 5:
                            # Retrieve skill filepaths in this namespace
                            async with db.execute("SELECT filepath FROM skills WHERE namespace = ?", (ns,)) as cursor:
                                skill_rows = await cursor.fetchall()
                            filepaths = [r[0] for r in skill_rows]
                            
                            # Cache the bundle
                            await db.execute("""
                                INSERT OR REPLACE INTO hotbar_cache (namespace, skills_list, updated_at)
                                VALUES (?, ?, CURRENT_TIMESTAMP)
                            """, (ns, json.dumps(filepaths)))
                            logger.info(f"Overseer: Cached hotbar bundle for namespace '{ns}'")
                            
                await db.commit()

        except Exception as e:
            logger.error(f"Overseer encountered an error: {e}")

        # Sleep and yield control to the event loop
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    logger.info("Overseer daemon shutting down cleanly.")

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
    from src.db import get_db
    logger.info("Overseer daemon started.")
    idle_threshold_seconds = 600

    while not shutdown_event.is_set():
        try:
            db = get_db()
            if db:
                # 1. Clean up idle sessions
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
                    
                    # Automatic Git Commits on Cold Memory
                    try:
                        proc = await asyncio.create_subprocess_shell(
                            f'git add . && git commit -m "Auto-commit session {session_id}"',
                            cwd=str(vault_path),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await proc.communicate()
                        if proc.returncode == 0:
                            logger.info(f"Overseer: Auto-committed cold memory for session {session_id}")
                        else:
                            logger.debug(f"Overseer: Auto-commit skipped or failed for {session_id}. {stderr.decode()}")
                    except Exception as e:
                        logger.error(f"Overseer: Git commit failed for {session_id}: {e}")

                await db.commit()

                # 2. Ranking & Bundling (Update Hotbar Cache)
                # Find namespaces where:
                # - Total unique skills used >= 5
                # - Used across >= 10 sessions
                # - Used across >= 5 distinct days
                # Rank by total usage.
                eligible_query = """
                    SELECT 
                        s.namespace,
                        COUNT(DISTINCT l.session_id) as session_count,
                        COUNT(DISTINCT date(l.timestamp)) as day_count,
                        COUNT(DISTINCT l.skill_name) as skill_count,
                        COUNT(l.id) as total_usage
                    FROM skill_usage_logs l
                    JOIN skills s ON l.skill_name = s.name
                    GROUP BY s.namespace
                    HAVING session_count >= 10 AND day_count >= 5 AND skill_count >= 5
                    ORDER BY total_usage DESC
                """
                async with db.execute(eligible_query) as cursor:
                    eligible_namespaces = await cursor.fetchall()
                
                # Keep top 5 only
                top_5 = eligible_namespaces[:5]
                top_5_namespaces = [row[0] for row in top_5]
                
                # Update hotbar cache for top 5 namespaces
                for ns, session_count, day_count, skill_count, total_usage in top_5:
                    # Retrieve skill filepaths in this namespace
                    async with db.execute("SELECT filepath FROM skills WHERE namespace = ?", (ns,)) as cursor:
                        skill_rows = await cursor.fetchall()
                    filepaths = [r[0] for r in skill_rows]
                    
                    # Cache the bundle
                    await db.execute("""
                        INSERT OR REPLACE INTO hotbar_cache (namespace, skills_list, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (ns, json.dumps(filepaths)))
                    logger.info(f"Overseer: Cached hotbar bundle for namespace '{ns}' (usage: {total_usage})")
                
                # Evict/delete any namespaces from hotbar_cache that are not in the top 5 anymore
                if top_5_namespaces:
                    placeholders = ",".join("?" for _ in top_5_namespaces)
                    await db.execute(f"""
                        DELETE FROM hotbar_cache 
                        WHERE namespace NOT IN ({placeholders})
                    """, top_5_namespaces)
                else:
                    await db.execute("DELETE FROM hotbar_cache")
                    
                await db.commit()

        except Exception as e:
            logger.error(f"Overseer encountered an error: {e}")

        # Sleep and yield control to the event loop
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    logger.info("Overseer daemon shutting down cleanly.")

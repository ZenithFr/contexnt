import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_parent_git(start_path: Path) -> Path | None:
    """
    Traverses upwards to detect a parent .git directory.
    Strictly stops at the user's home directory (~) to prevent modifying dotfiles.
    """
    home_dir = Path.home()
    current = start_path.resolve()
    
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        if current == home_dir:
            break
        current = current.parent
    return None

def init_obsidian_vault(vault_path: Path):
    """Initializes the Obsidian vault directory and safely handles git."""
    vault_path.mkdir(parents=True, exist_ok=True)
    
    parent_git = detect_parent_git(vault_path)
    if parent_git:
        if parent_git == Path.home():
            logger.warning(f"Detected root Git at home directory (~). Skipping gitignore modification to protect dotfiles.")
            return
            
        gitignore_path = parent_git / ".gitignore"
        relative_path = vault_path.relative_to(parent_git)
        
        try:
            ignore_entry = f"/{relative_path}/"
            if gitignore_path.exists():
                content = gitignore_path.read_text()
                if ignore_entry not in content:
                    with gitignore_path.open("a") as f:
                        f.write(f"\n{ignore_entry}\n")
            else:
                with gitignore_path.open("w") as f:
                    f.write(f"{ignore_entry}\n")
            logger.info(f"Appended {ignore_entry} to {gitignore_path}")
        except Exception as e:
            logger.error(f"Failed to modify parent .gitignore: {e}")

async def write_context_payload(vault_path: Path, session_id: str, payload: str) -> Path:
    """Writes the active context payload to the Obsidian vault."""
    file_path = vault_path / f"active_context_{session_id}.md"
    # Ensure the directory exists
    vault_path.mkdir(parents=True, exist_ok=True)
    
    file_path.write_text(payload, encoding="utf-8")
    return file_path

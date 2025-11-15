from pathlib import Path
import os

# Where the SQLite DB for experiments will live.
# You can override via env var: LLM_LAB_DB_PATH
DEFAULT_DB_PATH = Path(os.getenv("LLM_LAB_DB_PATH", "llm_lab.db"))

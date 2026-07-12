"""Load GEMINI_API_KEY and other settings from project .env file."""

from __future__ import annotations

import os


def load_project_env(project_dir: str) -> None:
    env_path = os.path.join(project_dir, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # Minimal parser if python-dotenv is not installed yet.
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value

"""Launch the MausamSetu FastAPI development server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    host = os.environ.get("MAUSAMSETU_API_HOST", "127.0.0.1")
    port = int(os.environ.get("MAUSAMSETU_API_PORT", "8011"))
    reload_enabled = os.environ.get("MAUSAMSETU_API_RELOAD", "0").lower() in {
        "1",
        "true",
        "yes",
    }

    print(
        "Starting MausamSetu API\n"
        f"  URL: http://{host}:{port}\n"
        f"  Live probe: http://{host}:{port}/livez\n"
        f"  Readiness: http://{host}:{port}/api/v1/health\n"
        f"  Reload: {'enabled' if reload_enabled else 'disabled'}",
        flush=True,
    )
    uvicorn.run(
        "mausamsetu.dashboard.api.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        reload_dirs=[str(ROOT / "mausamsetu")] if reload_enabled else None,
        log_level="info",
    )

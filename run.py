"""
CAD Material Tracker - Single entry point (dev and production).
  Local:  python run.py --reload
  Prod:   python run.py   (systemd runs this; no --reload)
WebSocket: websockets-sansio, no server-side pings; app handles heartbeats.
"""

import sys
import uvicorn


if __name__ == "__main__":
    reload_mode = "--reload" in sys.argv

    
    # Note: Scheduler starts automatically in FastAPI startup event
    uvicorn.run(
        "app.main:app",  # String import path (required for --reload)
        host="0.0.0.0", 
        port=8000,
        ws="websockets-sansio",  # Sans-IO impl: no server-side keepalive pings
        ws_ping_interval=None,   # Belt-and-suspenders: disable pings if ever read
        ws_ping_timeout=None,    # Belt-and-suspenders: disable ping timeout
        reload=reload_mode,
        reload_dirs=["app", "templates"] if reload_mode else None,  # Only watch app code, not venv/uploads
    )

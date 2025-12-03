#!/usr/bin/env python3
"""
Custom uvicorn server that disables WebSocket ping/pong timeout
We handle our own heartbeat in the application layer
"""
import uvicorn
from app import app

if __name__ == "__main__":
    # Configure uvicorn to run without websocket ping/pong
    # This prevents uvicorn from killing connections due to ping timeout
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        # Disable compression for faster transmission
        ws="websockets",  # Use websockets implementation
    )
    
    # Monkey-patch the websockets serve function to disable ping
    import websockets.server
    original_serve = websockets.server.serve
    
    def serve_no_ping(*args, **kwargs):
        # Disable ping/pong by setting ping_interval=None
        kwargs['ping_interval'] = None
        kwargs['ping_timeout'] = None
        return original_serve(*args, **kwargs)
    
    websockets.server.serve = serve_no_ping
    
    server = uvicorn.Server(config)
    print("[SERVER] Starting with WebSocket ping/pong DISABLED")
    print("[SERVER] Using application-level heartbeat instead")
    server.run()


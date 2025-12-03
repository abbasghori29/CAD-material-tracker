#!/usr/bin/env python3
"""
Custom uvicorn server that disables WebSocket ping/pong timeout
We handle our own heartbeat in the application layer
"""
import uvicorn
import sys
from app import app

if __name__ == "__main__":
    # Get port from command line or use default
    port = 8000
    if '--port' in sys.argv:
        port = int(sys.argv[sys.argv.index('--port') + 1])
    
    # Configure uvicorn to run without websocket ping/pong
    # This prevents uvicorn from killing connections due to ping timeout
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
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
        kwargs['max_size'] = 100 * 1024 * 1024  # 100MB max message size
        kwargs['max_queue'] = 1000  # Large queue for backpressure
        kwargs['write_limit'] = 100 * 1024 * 1024  # 100MB write buffer
        return original_serve(*args, **kwargs)
    
    websockets.server.serve = serve_no_ping
    
    server = uvicorn.Server(config)
    print(f"[SERVER] Starting on port {port} with WebSocket ping/pong DISABLED")
    print("[SERVER] Using application-level heartbeat instead")
    server.run()


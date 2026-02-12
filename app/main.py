"""
CAD Material Tracker - FastAPI Web Application
Main application initialization
"""

# Enable nested event loops for LlamaParse
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("[INIT] nest_asyncio applied - nested event loops enabled")
except ImportError:
    print("[INIT] WARNING: nest_asyncio not installed - LlamaParse may fail")
    print("[INIT] Install with: pip install nest-asyncio")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Import routers
from app.api.routes.home import router as home_router
from app.api.routes.upload import router as upload_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.cleanup import router as cleanup_router
from app.api.routes.auth import router as auth_router
from app.api.websocket import router as websocket_router

# Import startup components
from app.services.cleanup_service import start_scheduled_cleanup


# Create FastAPI app
app = FastAPI(title="CAD Material Tracker")

# Security headers middleware (add first so it runs last)
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import ALLOWED_ORIGINS_STR

# In production, set ALLOWED_ORIGINS env (e.g. https://yourdomain.com) and restrict below
_cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if ALLOWED_ORIGINS_STR:
    _cors_origins = [o.strip() for o in ALLOWED_ORIGINS_STR.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_images", StaticFiles(directory="output_images"), name="output_images")

# Include routers
app.include_router(home_router)
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(jobs_router)
app.include_router(cleanup_router)
app.include_router(websocket_router)


@app.on_event("startup")
async def startup_event():
    """Start scheduled tasks on application startup"""
    start_scheduled_cleanup()
    print("[STARTUP] Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[SHUTDOWN] Application shutting down")

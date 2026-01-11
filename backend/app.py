"""
EvalX Backend Application
=========================

Main FastAPI application entry point. Configures CORS, mounts routers,
and handles application lifecycle.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import route modules
from routes.auth import router as auth_router
from routes.developer import router as developer_router
from routes.dashboard import router as dashboard_router
from routes.team import router as team_router
from routes.connect import router as connect_router
from routes.interview import router as interview_router
from routes.analytics import router as analytics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    logger.info("EvalX API starting up...")
    yield
    logger.info("EvalX API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="EvalX API",
    description="AI-Powered Hackathon Evaluation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(auth_router, prefix="/api/auth")
app.include_router(developer_router, prefix="/api/developer")
app.include_router(dashboard_router, prefix="/api/dashboard")
app.include_router(team_router, prefix="/api/team")
app.include_router(connect_router, prefix="/api/connect")
app.include_router(interview_router, prefix="/api/interview")

# Mount analytics router (NEW - Advanced Analytics Dashboard)
app.include_router(analytics_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "success": True,
        "message": "EvalX API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "evalx-api"
    }

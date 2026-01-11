"""
EvalX Backend Application
=========================
Main FastAPI application with all routes registered (Merged khushi + main).
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import common routers
from routes.auth import router as auth_router
from routes.connect import router as connect_router
from routes.dashboard import router as dashboard_router
from routes.developer import router as developer_router
from routes.interview import router as interview_router
from routes.team import router as team_router

# Import MAIN branch routers
from routes.ai_models.create_event import router as create_event_router
from routes.domain_evaluation import router as domain_evaluation_router
from graph.github import router as github_router
from graph.ppt_evaluator import router as ppt_router

# Import KHUSHI branch routers (New Features)
from routes.analytics import router as analytics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EvalX API starting up...")
    yield
    logger.info("EvalX API shutting down...")

# Create FastAPI app
app = FastAPI(
    title="EvalX API",
    description="AI-Powered Hackathon Evaluation Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register All Routers ---

# Standard Routes
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(connect_router, prefix="/api/connect", tags=["Connect"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(developer_router, prefix="/api/developer", tags=["Developer"])
app.include_router(interview_router, prefix="/api/interview", tags=["Interview"])
app.include_router(team_router, prefix="/api/team", tags=["Team"])

# Main Branch Exclusive Routes
app.include_router(create_event_router, prefix="/api/ai", tags=["AI Models"])
app.include_router(github_router, prefix="/api/github", tags=["GitHub"])
app.include_router(ppt_router, prefix="/api/ppt", tags=["PPT Evaluator"])
app.include_router(domain_evaluation_router, prefix="/api/domain-evaluation", tags=["Domain Evaluation"])

# Khushi Branch Exclusive Routes (Analytics)
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "EvalX API is running", "version": "1.0.0"}

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {"domain_evaluators": "active", "analytics": "active"},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
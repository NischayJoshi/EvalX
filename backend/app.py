"""
EvalX Backend - FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import auth, team, dashboard, developer, connect, interview
from routes import async_submissions, websocket

app = FastAPI(
    title="EvalX API",
    description="AI-powered evaluation platform for hackathons and coding events",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(team.router, prefix="/api/teams", tags=["Teams"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(developer.router, prefix="/api/developer", tags=["Developer"])
app.include_router(connect.router, prefix="/api/connect", tags=["Connect"])
app.include_router(interview.router, prefix="/api/interview", tags=["Interview"])
app.include_router(async_submissions.router, prefix="/api/submissions", tags=["Async Submissions"])
app.include_router(websocket.router, tags=["WebSocket"])



@app.get("/")
async def root():
    return {"message": "EvalX API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

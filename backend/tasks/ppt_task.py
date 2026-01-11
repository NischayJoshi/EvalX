import os
import json
import tempfile
import asyncio
from typing import Dict, Any, Optional
from celery import current_task
from celery_app import celery_app
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["evalx"]
submissions_collection = db["submissions"]


def run_async(coro):
    """Helper to run async code in sync Celery task"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def update_task_progress(submission_id: str, stage: str, progress: int, message: str):
    """Update task progress for WebSocket notifications"""
    current_task.update_state(
        state="PROGRESS",
        meta={
            "submission_id": submission_id,
            "stage": stage,
            "progress": progress,
            "message": message,
        }
    )


async def _update_submission_status(submission_id: str, status: str, data: Optional[Dict] = None):
    """Update submission status in database"""
    update_doc = {
        "status": status,
        "updatedAt": datetime.utcnow(),
    }
    if data:
        update_doc["aiResult"] = data
    
    await submissions_collection.update_one(
        {"_id": ObjectId(submission_id)},
        {"$set": update_doc}
    )


async def _evaluate_ppt(submission_id: str, file_url: str, topic: str) -> Dict[str, Any]:
    """Core PPT evaluation logic"""
    from graph.ppt_evaluator import analyze_ppt_with_gpt, State
    
    update_task_progress(submission_id, "extracting", 10, "Extracting slides from PPT...")
    
    state: State = {
        "mode": "ppt",
        "content": topic,
        "file_path": file_url,
        "github_url": None,
        "video_url": None,
        "output": None,
    }
    
    update_task_progress(submission_id, "analyzing", 30, "Analyzing slides with AI...")
    
    result_state = await analyze_ppt_with_gpt(state)
    
    update_task_progress(submission_id, "scoring", 70, "Computing scores...")
    
    output = result_state.get("output", {})
    
    update_task_progress(submission_id, "finalizing", 90, "Generating feedback...")
    
    return output


@celery_app.task(bind=True, name="tasks.ppt_task.evaluate_ppt_task")
def evaluate_ppt_task(self, submission_id: str, file_url: str, topic: str, event_id: str, team_id: str):
    """
    Celery task to evaluate PPT submission asynchronously.
    
    Args:
        submission_id: MongoDB submission document ID
        file_url: Cloudinary URL of uploaded PPT
        topic: Event/project topic for context
        event_id: Event ID
        team_id: Team ID
    """
    try:
        update_task_progress(submission_id, "queued", 5, "Starting PPT evaluation...")
        
        async def process():
            await _update_submission_status(submission_id, "processing")
            
            result = await _evaluate_ppt(submission_id, file_url, topic)
            
            await _update_submission_status(submission_id, "completed", result)
            
            return result
        
        result = run_async(process())
        
        update_task_progress(submission_id, "completed", 100, "Evaluation complete!")
        
        return {
            "success": True,
            "submission_id": submission_id,
            "result": result,
        }
        
    except Exception as e:
        error_msg = str(e)
        
        async def mark_error():
            await _update_submission_status(submission_id, "error", {"error": error_msg})
        
        run_async(mark_error())
        
        update_task_progress(submission_id, "error", 0, f"Evaluation failed: {error_msg}")
        
        raise self.retry(exc=e, countdown=60, max_retries=2)

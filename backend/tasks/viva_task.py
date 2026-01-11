import os
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
viva_sessions_collection = db["viva_sessions"]


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
        update_doc["vivaResult"] = data
    
    await submissions_collection.update_one(
        {"_id": ObjectId(submission_id)},
        {"$set": update_doc}
    )


@celery_app.task(bind=True, name="tasks.viva_task.process_viva_task")
def process_viva_task(self, submission_id: str, session_id: str, event_id: str, team_id: str):
    """
    Celery task to process viva/interview session results.
    
    This task is triggered after a viva session is completed to:
    1. Aggregate all Q&A scores
    2. Generate final viva report
    3. Update submission with results
    
    Args:
        submission_id: MongoDB submission document ID
        session_id: Viva session ID
        event_id: Event ID
        team_id: Team ID
    """
    try:
        update_task_progress(submission_id, "processing", 10, "Processing viva results...")
        
        async def process():
            await _update_submission_status(submission_id, "processing")
            
            session = await viva_sessions_collection.find_one({"_id": ObjectId(session_id)})
            
            if not session:
                raise ValueError(f"Viva session {session_id} not found")
            
            update_task_progress(submission_id, "scoring", 50, "Calculating viva scores...")
            
            answers = session.get("answers", [])
            total_score = 0
            max_score = 0
            
            for ans in answers:
                score = ans.get("score", 0)
                total_score += score
                max_score += 10
            
            final_score = (total_score / max_score * 100) if max_score > 0 else 0
            
            result = {
                "session_id": session_id,
                "total_questions": len(answers),
                "total_score": total_score,
                "max_score": max_score,
                "final_score": round(final_score, 2),
                "answers": answers,
                "completed_at": datetime.utcnow().isoformat(),
            }
            
            update_task_progress(submission_id, "finalizing", 90, "Saving viva results...")
            
            await _update_submission_status(submission_id, "completed", result)
            
            return result
        
        result = run_async(process())
        
        update_task_progress(submission_id, "completed", 100, "Viva evaluation complete!")
        
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
        
        update_task_progress(submission_id, "error", 0, f"Viva processing failed: {error_msg}")
        
        raise self.retry(exc=e, countdown=30, max_retries=2)

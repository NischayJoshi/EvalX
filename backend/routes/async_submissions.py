"""
Async Submission Routes - Uses Celery for background processing
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from middlewares.auth_required import get_user as get_current_user
from config.db import db
from bson import ObjectId
from datetime import datetime
from typing import Optional
import cloudinary.uploader
from utils.serializers import serialize_doc

router = APIRouter()

users_collection = db["users"]
events_collection = db["events"]
teams_collection = db["teams"]
submissions_collection = db["submissions"]


async def get_user_team(event_id: str, user_id: str):
    """Get user's team for an event"""
    user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_doc:
        raise HTTPException(404, "User not found")
    
    team = await teams_collection.find_one({
        "eventId": event_id,
        "members.userId": str(user_doc["_id"])
    })
    
    if not team:
        raise HTTPException(403, "You must join a team first")
    
    if not team.get("isActive"):
        raise HTTPException(403, "Team is not active yet (needs minimum members)")
    
    return team


@router.post("/events/{event_id}/submit/ppt/async")
async def submit_ppt_async(
    event_id: str,
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Submit PPT for evaluation - processes asynchronously via Celery.
    Returns immediately with submission ID for tracking.
    """
    from tasks.ppt_task import evaluate_ppt_task
    
    team = await get_user_team(event_id, user["id"])
    team_id = str(team["_id"])
    
    existing = await submissions_collection.find_one({
        "eventId": event_id,
        "teamId": team_id,
        "roundId": "ppt"
    })
    if existing:
        raise HTTPException(400, "PPT already submitted for this round")
    
    event = await events_collection.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(404, "Event not found")
    
    upload_result = cloudinary.uploader.upload(
        file.file,
        resource_type="raw",
        folder=f"evalx/submissions/{event_id}/{team_id}"
    )
    file_url = upload_result["secure_url"]
    
    submission_doc = {
        "eventId": event_id,
        "teamId": team_id,
        "roundId": "ppt",
        "fileUrl": file_url,
        "status": "queued",
        "submittedAt": datetime.utcnow(),
    }
    
    result = await submissions_collection.insert_one(submission_doc)
    submission_id = str(result.inserted_id)
    
    topic = event.get("name", "") + " - " + event.get("summary", "")
    
    task = evaluate_ppt_task.delay(
        submission_id=submission_id,
        file_url=file_url,
        topic=topic,
        event_id=event_id,
        team_id=team_id
    )
    
    await submissions_collection.update_one(
        {"_id": result.inserted_id},
        {"$set": {"celeryTaskId": task.id}}
    )
    
    return {
        "success": True,
        "message": "PPT submitted for evaluation",
        "submissionId": submission_id,
        "taskId": task.id,
        "status": "queued"
    }


@router.post("/events/{event_id}/submit/repo/async")
async def submit_repo_async(
    event_id: str,
    repo: str = Form(...),
    video: str = Form(...),
    user=Depends(get_current_user)
):
    """
    Submit GitHub repo for evaluation - processes asynchronously via Celery.
    Returns immediately with submission ID for tracking.
    """
    from tasks.github_task import evaluate_github_task
    
    team = await get_user_team(event_id, user["id"])
    team_id = str(team["_id"])
    
    existing = await submissions_collection.find_one({
        "eventId": event_id,
        "teamId": team_id,
        "roundId": "repo"
    })
    if existing:
        raise HTTPException(400, "Repository already submitted for this round")
    
    event = await events_collection.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(404, "Event not found")
    
    submission_doc = {
        "eventId": event_id,
        "teamId": team_id,
        "roundId": "repo",
        "repo": repo,
        "video": video,
        "status": "queued",
        "submittedAt": datetime.utcnow(),
    }
    
    result = await submissions_collection.insert_one(submission_doc)
    submission_id = str(result.inserted_id)
    
    project_desc = event.get("description", "") or event.get("summary", "")
    
    task = evaluate_github_task.delay(
        submission_id=submission_id,
        repo_url=repo,
        video_url=video,
        project_desc=project_desc,
        event_id=event_id,
        team_id=team_id
    )
    
    await submissions_collection.update_one(
        {"_id": result.inserted_id},
        {"$set": {"celeryTaskId": task.id}}
    )
    
    return {
        "success": True,
        "message": "Repository submitted for evaluation",
        "submissionId": submission_id,
        "taskId": task.id,
        "status": "queued"
    }


@router.get("/submissions/{submission_id}/status")
async def get_submission_status(
    submission_id: str,
    user=Depends(get_current_user)
):
    """
    Get the current status of a submission.
    Useful for polling until evaluation completes.
    """
    from celery.result import AsyncResult
    from celery_app import celery_app
    
    submission = await submissions_collection.find_one({"_id": ObjectId(submission_id)})
    
    if not submission:
        raise HTTPException(404, "Submission not found")
    
    response = {
        "submissionId": submission_id,
        "status": submission.get("status", "unknown"),
        "roundId": submission.get("roundId"),
        "submittedAt": submission.get("submittedAt"),
    }
    
    task_id = submission.get("celeryTaskId")
    if task_id:
        task_result = AsyncResult(task_id, app=celery_app)
        
        if task_result.state == "PROGRESS":
            meta = task_result.info or {}
            response["progress"] = {
                "stage": meta.get("stage"),
                "percent": meta.get("progress"),
                "message": meta.get("message"),
            }
        elif task_result.state == "SUCCESS":
            response["status"] = "completed"
        elif task_result.state == "FAILURE":
            response["status"] = "error"
            response["error"] = str(task_result.result)
    
    if submission.get("status") == "completed":
        if submission.get("roundId") == "ppt":
            response["result"] = submission.get("aiResult")
        elif submission.get("roundId") == "repo":
            response["result"] = submission.get("evaluation")
    
    return {"success": True, "data": response}


@router.get("/submissions/{submission_id}/result")
async def get_submission_result(
    submission_id: str,
    user=Depends(get_current_user)
):
    """
    Get the full evaluation result for a completed submission.
    """
    submission = await submissions_collection.find_one({"_id": ObjectId(submission_id)})
    
    if not submission:
        raise HTTPException(404, "Submission not found")
    
    if submission.get("status") != "completed":
        raise HTTPException(400, f"Submission not completed yet. Status: {submission.get('status')}")
    
    return {
        "success": True,
        "data": serialize_doc(submission)
    }

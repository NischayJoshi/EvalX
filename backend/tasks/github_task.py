import os
import json
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
        update_doc["evaluation"] = data
    
    await submissions_collection.update_one(
        {"_id": ObjectId(submission_id)},
        {"$set": update_doc}
    )


async def _evaluate_github(submission_id: str, repo_url: str, video_url: str, project_desc: str) -> Dict[str, Any]:
    """Core GitHub repository evaluation logic with caching support"""
    from graph.github import (
        clone_repo,
        get_code_chunks,
        static_analysis,
        plagiarism_score,
        analyze_structure,
        detect_code_smells,
        compute_risk_score,
        llm_code_rating,
        compute_final_score,
        rubric_from_score,
        generate_markdown_mentor,
        generate_rewrite_suggestions,
        generate_pdf_report,
        safe_rmtree,
    )
    from utils.cache import Cache, get_latest_commit_hash
    import base64
    
    update_task_progress(submission_id, "checking_cache", 5, "Checking cache...")
    
    commit_hash = get_latest_commit_hash(repo_url)
    if commit_hash:
        cached_result = Cache.get_repo_evaluation(repo_url, commit_hash)
        if cached_result:
            update_task_progress(submission_id, "cache_hit", 100, "Using cached evaluation!")
            return cached_result
    
    update_task_progress(submission_id, "cloning", 10, "Cloning repository...")
    
    repo_path = clone_repo(repo_url)
    
    try:
        update_task_progress(submission_id, "scanning", 20, "Scanning code files...")
        chunks = get_code_chunks(repo_path)
        
        update_task_progress(submission_id, "static_analysis", 30, "Running static analysis (Pylint, Radon)...")
        radon_raw, pylint_score = static_analysis(repo_path)
        
        update_task_progress(submission_id, "plagiarism", 40, "Checking for code duplication...")
        plag = plagiarism_score(repo_path)
        
        update_task_progress(submission_id, "structure", 50, "Analyzing project structure...")
        structure = analyze_structure(repo_path)
        
        update_task_progress(submission_id, "ai_review", 60, "AI reviewing code quality...")
        logic, rel, style, llm_fb = await llm_code_rating(project_desc, chunks)
        
        update_task_progress(submission_id, "scoring", 70, "Computing scores and detecting code smells...")
        code_smells = detect_code_smells(radon_raw, pylint_score, plag, structure)
        risk_score = compute_risk_score(plag, pylint_score, code_smells, structure)
        final_score = compute_final_score(plag, logic, rel, style, pylint_score, structure)
        rubric = rubric_from_score(final_score)
        
        update_task_progress(submission_id, "feedback", 80, "Generating detailed feedback...")
        
        result = {
            "final_score": final_score,
            "rubric": rubric,
            "risk_score": risk_score,
            "structure": structure,
            "plagiarism": plag,
            "logic": logic,
            "relevance": rel,
            "style": style,
            "pylint_score": pylint_score,
            "code_smells": code_smells,
            "llm_feedback": llm_fb,
            "files_analyzed": len(chunks),
        }
        
        update_task_progress(submission_id, "mentor", 85, "Creating mentor summary...")
        mentor_md = await generate_markdown_mentor(project_desc, result)
        result["mentor_summary_markdown"] = mentor_md
        
        update_task_progress(submission_id, "rewrite", 90, "Generating rewrite suggestions...")
        rewrite_md = await generate_rewrite_suggestions(project_desc, chunks, code_smells)
        result["rewrite_suggestions_markdown"] = rewrite_md
        
        update_task_progress(submission_id, "pdf", 95, "Generating PDF report...")
        pdf_bytes = generate_pdf_report(result)
        result["report_pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")
        
        if commit_hash:
            Cache.set_repo_evaluation(repo_url, commit_hash, result)
        
        return result
        
    finally:
        safe_rmtree(repo_path)


@celery_app.task(bind=True, name="tasks.github_task.evaluate_github_task")
def evaluate_github_task(self, submission_id: str, repo_url: str, video_url: str, project_desc: str, event_id: str, team_id: str):
    """
    Celery task to evaluate GitHub repository submission asynchronously.
    
    Args:
        submission_id: MongoDB submission document ID
        repo_url: GitHub repository URL
        video_url: Demo video URL
        project_desc: Project description for context
        event_id: Event ID
        team_id: Team ID
    """
    try:
        update_task_progress(submission_id, "queued", 5, "Starting repository evaluation...")
        
        async def process():
            await _update_submission_status(submission_id, "processing")
            
            result = await _evaluate_github(submission_id, repo_url, video_url, project_desc)
            
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
        
        raise self.retry(exc=e, countdown=120, max_retries=2)

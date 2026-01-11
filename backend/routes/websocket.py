"""
WebSocket Routes - Real-time updates for submissions
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Set
import asyncio
import json
from celery.result import AsyncResult
from celery_app import celery_app
from config.db import db
from bson import ObjectId

router = APIRouter()

submissions_collection = db["submissions"]


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, submission_id: str):
        await websocket.accept()
        if submission_id not in self.active_connections:
            self.active_connections[submission_id] = set()
        self.active_connections[submission_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, submission_id: str):
        if submission_id in self.active_connections:
            self.active_connections[submission_id].discard(websocket)
            if not self.active_connections[submission_id]:
                del self.active_connections[submission_id]
    
    async def broadcast_to_submission(self, submission_id: str, message: dict):
        if submission_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[submission_id]:
                try:
                    await connection.send_json(message)
                except:
                    dead_connections.add(connection)
            
            for conn in dead_connections:
                self.active_connections[submission_id].discard(conn)


manager = ConnectionManager()


async def poll_celery_task(submission_id: str, task_id: str, websocket: WebSocket):
    """Poll Celery task status and send updates via WebSocket"""
    last_state = None
    last_progress = None
    
    while True:
        try:
            task_result = AsyncResult(task_id, app=celery_app)
            state = task_result.state
            
            if state == "PENDING":
                message = {
                    "type": "status",
                    "submissionId": submission_id,
                    "status": "queued",
                    "message": "Waiting in queue..."
                }
            elif state == "PROGRESS":
                meta = task_result.info or {}
                progress = meta.get("progress", 0)
                
                if progress != last_progress:
                    message = {
                        "type": "progress",
                        "submissionId": submission_id,
                        "status": "processing",
                        "stage": meta.get("stage"),
                        "progress": progress,
                        "message": meta.get("message", "Processing...")
                    }
                    last_progress = progress
                else:
                    await asyncio.sleep(1)
                    continue
            elif state == "SUCCESS":
                submission = await submissions_collection.find_one(
                    {"_id": ObjectId(submission_id)}
                )
                message = {
                    "type": "completed",
                    "submissionId": submission_id,
                    "status": "completed",
                    "message": "Evaluation complete!",
                    "hasResult": True
                }
                await websocket.send_json(message)
                return
            elif state == "FAILURE":
                message = {
                    "type": "error",
                    "submissionId": submission_id,
                    "status": "error",
                    "message": str(task_result.result),
                }
                await websocket.send_json(message)
                return
            else:
                message = {
                    "type": "status",
                    "submissionId": submission_id,
                    "status": state.lower(),
                    "message": f"Status: {state}"
                }
            
            if state != last_state or state == "PROGRESS":
                await websocket.send_json(message)
                last_state = state
            
            await asyncio.sleep(1)
            
        except WebSocketDisconnect:
            return
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass
            return


@router.websocket("/ws/submissions/{submission_id}")
async def websocket_submission_status(
    websocket: WebSocket,
    submission_id: str
):
    """
    WebSocket endpoint for real-time submission status updates.
    
    Connect to receive live progress updates as the evaluation runs.
    Messages sent include:
    - status: General status updates
    - progress: Progress percentage and stage info
    - completed: Final completion with results available
    - error: Error occurred during processing
    """
    await manager.connect(websocket, submission_id)
    
    try:
        submission = await submissions_collection.find_one(
            {"_id": ObjectId(submission_id)}
        )
        
        if not submission:
            await websocket.send_json({
                "type": "error",
                "message": "Submission not found"
            })
            await websocket.close()
            return
        
        if submission.get("status") == "completed":
            await websocket.send_json({
                "type": "completed",
                "submissionId": submission_id,
                "status": "completed",
                "message": "Evaluation already complete",
                "hasResult": True
            })
            return
        
        task_id = submission.get("celeryTaskId")
        
        if task_id:
            await poll_celery_task(submission_id, task_id, websocket)
        else:
            await websocket.send_json({
                "type": "status",
                "submissionId": submission_id,
                "status": submission.get("status", "unknown"),
                "message": "No active task found"
            })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "status":
                    submission = await submissions_collection.find_one(
                        {"_id": ObjectId(submission_id)}
                    )
                    await websocket.send_json({
                        "type": "status",
                        "submissionId": submission_id,
                        "status": submission.get("status") if submission else "unknown"
                    })
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, submission_id)


@router.websocket("/ws/events/{event_id}/submissions")
async def websocket_event_submissions(
    websocket: WebSocket,
    event_id: str
):
    """
    WebSocket endpoint for real-time updates on all submissions in an event.
    Useful for organizers monitoring multiple submissions.
    """
    await websocket.accept()
    
    try:
        while True:
            submissions = await submissions_collection.find({
                "eventId": event_id
            }).to_list(length=100)
            
            stats = {
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "error": 0
            }
            
            for sub in submissions:
                status = sub.get("status", "unknown")
                if status in stats:
                    stats[status] += 1
            
            await websocket.send_json({
                "type": "event_stats",
                "eventId": event_id,
                "totalSubmissions": len(submissions),
                "stats": stats
            })
            
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        pass

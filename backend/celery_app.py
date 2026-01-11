from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "evalx_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "tasks.ppt_task",
        "tasks.github_task",
        "tasks.viva_task",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,  # Fair distribution
    worker_concurrency=4,  # 4 parallel workers
)

celery_app.conf.task_routes = {
    "tasks.ppt_task.*": {"queue": "ppt_queue"},
    "tasks.github_task.*": {"queue": "github_queue"},
    "tasks.viva_task.*": {"queue": "viva_queue"},
}

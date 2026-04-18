"""backend/worker/celery_app.py — Celery with CloudAMQP broker, no result backend."""
from celery import Celery
from backend.config import settings


celery_app = Celery(
    "cineagent_worker",
    broker=settings.celery_broker_url,  # CloudAMQP (RabbitMQ)
    backend=None,                        # fire-and-forget — results not needed
    include=["backend.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=False,
    broker_connection_retry_on_startup=True,
)
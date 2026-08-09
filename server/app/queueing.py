from redis import Redis
from rq import Queue

from .config import get_settings


def training_queue() -> Queue:
    return Queue("training", connection=Redis.from_url(get_settings().redis_url), default_timeout=604_800)

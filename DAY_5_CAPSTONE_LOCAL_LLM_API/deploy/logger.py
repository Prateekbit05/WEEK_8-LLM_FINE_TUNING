"""
============================================================
 DAY 5 CAPSTONE — logger.py
 Structured JSON logging
============================================================
"""

import logging
import os
import uuid
import json
from datetime import datetime
from deploy.config import config

os.makedirs(config.log_dir, exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)


def setup_logger(name: str = "llm_api") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    file_handler = logging.FileHandler(
        os.path.join(config.log_dir, "api.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    error_handler = logging.FileHandler(
        os.path.join(config.log_dir, "errors.log")
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


logger = setup_logger()
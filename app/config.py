import logging

from pydantic import BaseModel


class LoggerConfig(BaseModel):
    """Logging configuration to be set for the server"""

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    }
    loggers: dict = {
        "app": {"handlers": ["default"], "level": logging.INFO},
        "uvicorn": {"handlers": ["default"], "level": logging.INFO},
        "uvicorn.error": {"level": logging.WARNING},
        "uvicorn.access": {"handlers": ["default"], "level": logging.WARNING},
    }

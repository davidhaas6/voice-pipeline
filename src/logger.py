import logging
import os
import sys

from dotenv import load_dotenv


def setup_logging():
    """
    Configures logging based on the LOG_LEVEL environment variable.
    Sets root logger to WARNING to avoid library noise, while
    letting 'src' and '__main__' modules log at the specified level.
    """
    load_dotenv()
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    print(f"LOG_LEVEL: {log_level_str}")
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Simple formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Root logger: set to WARNING to silence library noise by default
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Avoid adding multiple handlers if setup is called multiple times
    if not root.handlers:
        root.addHandler(handler)

    # Application loggers: set to user's desired level
    # 'src' handles all submodules like 'src.bot', 'src.tts'
    logging.getLogger("src").setLevel(log_level)
    # '__main__' handles the entry point script
    logging.getLogger("__main__").setLevel(log_level)


def get_logger(name):
    """
    Returns a logger for the given name.
    """
    return logging.getLogger(name)

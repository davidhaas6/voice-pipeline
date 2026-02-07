import hashlib
import logging
import os
import sys

from dotenv import load_dotenv


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds colors based on log level and logger name.
    """

    # ANSI Color Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }

    # A selection of distinct colors for module names
    NAME_COLORS = [
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]

    def _get_name_color(self, name):
        """Deterministically pick a color based on the logger name."""
        # Use a hash of the name to consistently pick the same color for the same module
        hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
        return self.NAME_COLORS[hash_val % len(self.NAME_COLORS)]

    def format(self, record):
        # Color the level name
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        levelname = f"{level_color}{record.levelname:8}{self.RESET}"

        # Color the logger name (module/class)
        name_color = self._get_name_color(record.name)
        name = f"{name_color}{record.name}{self.RESET}"

        # Dim the location info
        location = f"{self.DIM}({record.filename}:{record.lineno}){self.RESET}"

        # Format the timestamp
        asctime = self.formatTime(record, "%H:%M:%S")

        # Get the formatted message
        message = record.getMessage()

        # Handle exceptions if any
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        # Construct the final line
        formatted = f"{asctime} {levelname} {name} {location} - {message}"

        if record.exc_text:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + record.exc_text
        if record.stack_info:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + self.formatStack(record.stack_info)

        return formatted


def setup_logging():
    """
    Configures logging based on the LOG_LEVEL environment variable.
    Sets root logger to WARNING to avoid library noise, while
    letting 'src' and '__main__' modules log at the specified level.
    """
    load_dotenv()
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Console handler with the colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    # Root logger: set to WARNING to silence library noise by default
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Avoid adding multiple handlers if setup is called multiple times
    if not root.handlers:
        root.addHandler(handler)

    # Application loggers: set to user's desired level
    logging.getLogger("src").setLevel(log_level)
    logging.getLogger("__main__").setLevel(log_level)


def get_logger(name):
    """
    Returns a logger for the given name.
    """
    return logging.getLogger(name)

from src.logger import setup_logging

setup_logging()

from src.bot import run  # noqa: E402 (needs to be after setup_logging)

if __name__ == "__main__":
    run()

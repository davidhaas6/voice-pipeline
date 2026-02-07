from src.logger import setup_logging
from src.tts import TTSManager

# 1. Setup logging
setup_logging()

# 2. Pre-load the heavy TTS model before bot starts
TTSManager.preload()

from src.bot import run  # noqa: E402 (needs to be after setup_logging)

if __name__ == "__main__":
    run()

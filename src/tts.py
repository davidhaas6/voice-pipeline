import os
import queue
import random
import threading
import time

import numpy as np
from pocket_tts import TTSModel

from .logger import get_logger

logger = get_logger(__name__)


class TTSManager:
    _model = None  # Class-level shared model

    @staticmethod
    def preload():
        """Pre-loads the shared TTS model into memory."""
        if TTSManager._model is None:
            logger.info("[TTS] Pre-loading shared model...")
            TTSManager._model = TTSModel.load_model()
        return TTSManager._model

    def __init__(self, default_voice_path="data/voice/d_hermit.wav"):
        # Ensure model is loaded (either via preload or first init)
        self.model = TTSManager.preload()
        self.voices = {}
        self.sample_rate = self.model.sample_rate

        # Background worker setup
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        self.add_voice("narrator", default_voice_path)
        logger.info("Initialized TTS")

    def add_voice(self, alias: str, audio_source: str):
        """
        Extracts model state for a given audio file and stores it.
        """
        try:
            state = self.model.get_state_for_audio_prompt(audio_source, truncate=True)
            logger.info(f"Loaded voice '{alias}' from {audio_source}")
            self.voices[alias] = state
        except Exception as e:
            logger.error(f"Failed to load voice '{alias}': {e}")

    def speak(self, text: str, voice: str = "narrator", callback=None):
        """
        Queues text for generation and output via callback.
        """
        if not callback:
            raise ValueError("A callback must be provided for audio output.")

        if voice == "random":
            voice = random.choice(list(self.voices.keys()))
        self.queue.put((text, voice, callback))

    def stop(self):
        """
        Clears the queue and signals current generation to stop.
        """
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self.stop_event.set()

        logger.info("Playback stopped and queue cleared.")
        # Reset stop event for future calls
        # Note: stop_event is cleared at the start of _worker next iteration

    def _worker(self):
        """
        Background worker that processes the speech queue.
        """
        while True:
            text, voice, callback = self.queue.get()
            if voice not in self.voices:
                voice = "narrator" if "narrator" in self.voices else None

            if not voice:
                self.queue.task_done()
                continue

            state = self.voices[voice]
            self.stop_event.clear()

            logger.info(f'Generating: "{text}"')
            # all these are for performance logging
            start_time = time.perf_counter()
            first_chunk_time = None
            chunk_count = 0
            total_samples = 0

            try:
                for chunk in self.model.generate_audio_stream(state, text):
                    if self.stop_event.is_set():
                        break

                    now = time.perf_counter()  # for performance logging
                    if first_chunk_time is None:
                        first_chunk_time = now
                        logger.debug(
                            f"TTFB: {(first_chunk_time - start_time) * 1000:.2f}ms"
                        )

                    audio_data = chunk.cpu().numpy().astype(np.float32)

                    # for performance logging
                    chunk_count += 1
                    total_samples += len(audio_data)

                    callback(audio_data)

                # for performance logging
                end_time = time.perf_counter()

                if chunk_count > 0:
                    duration = total_samples / self.sample_rate
                    logger.info(
                        f"Done. Total time: {(end_time - start_time):.2f}s | Audio duration: {duration:.2f}s | Real-time factor: {duration / (end_time - start_time):.2f}x"
                    )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
            finally:
                self.queue.task_done()


if __name__ == "__main__":
    # Quick test if run directly
    import os
    import time

    voice_file = "data/narrator.wav"
    if os.path.exists(voice_file):
        from .logger import setup_logging

        setup_logging()
        manager = TTSManager(voice_file)
        logger.info("Speaking background...")

        def dummy_callback(data):
            logger.debug(f"Received chunk of size {len(data)}")

        manager.speak(
            "This is a test of the background voice system.", callback=dummy_callback
        )
        manager.speak(
            "This second sentence should play immediately after the first.",
            callback=dummy_callback,
        )
        time.sleep(2)
        logger.info("Stopping mid-speech...")
        manager.stop()
        time.sleep(1)
    else:
        logger.error(f"Test file {voice_file} not found.")

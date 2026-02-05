import glob
import os
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pocket_tts import TTSModel


class TTSManager:
    def __init__(self, default_voice_path="data/voice/d_hermit.wav"):
        # print("[TTS] Initializing model...")
        self.model = TTSModel.load_model()
        self.voices = {}
        self.sample_rate = self.model.sample_rate

        # Background worker setup
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # Add initial voices in parallel
        voice_files = glob.glob("data/voice/*.wav")
        voices_to_load = [
            (os.path.basename(f).split(".")[0], f)
            for f in voice_files
            if os.path.basename(f).split(".")[0]
            != os.path.basename(default_voice_path).split(".")[0]
        ]
        voices_to_load.append(("narrator", default_voice_path))
        voices_to_load = list(set(voices_to_load))

        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda x: self.add_voice(*x), voices_to_load))

    def add_voice(self, alias: str, audio_source: str):
        """
        Extracts model state for a given audio file and stores it.
        """
        try:
            state = self.model.get_state_for_audio_prompt(audio_source, truncate=True)
            print(f"[TTS] Loaded voice '{alias}' from {audio_source}")
            self.voices[alias] = state
        except Exception as e:
            print(f"[TTS] Failed to load voice '{alias}': {e}")

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

        print("[TTS] Playback stopped and queue cleared.")
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

            try:
                for chunk in self.model.generate_audio_stream(state, text):
                    if self.stop_event.is_set():
                        break

                    audio_data = chunk.cpu().numpy().astype(np.float32)
                    callback(audio_data)
            except Exception as e:
                print(f"[TTS] Error during generation: {e}")
            finally:
                self.queue.task_done()


if __name__ == "__main__":
    # Quick test if run directly
    import os
    import time

    voice_file = "data/narrator.wav"
    if os.path.exists(voice_file):
        manager = TTSManager(voice_file)
        print("Speaking background...")

        def dummy_callback(data):
            print(f"Received chunk of size {len(data)}")

        manager.speak(
            "This is a test of the background voice system.", callback=dummy_callback
        )
        manager.speak(
            "This second sentence should play immediately after the first.",
            callback=dummy_callback,
        )
        time.sleep(2)
        print("Stopping mid-speech...")
        manager.stop()
        time.sleep(1)
    else:
        print(f"Test file {voice_file} not found.")

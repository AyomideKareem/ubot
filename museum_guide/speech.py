from collections import deque
import threading
import time
from typing import Deque, Optional

from .hardware import HardwareCapabilityError, HardwareFault, MuseumHardware


class SpeechQueue:
    def __init__(self, hardware: MuseumHardware, timeout_s: float = 5.0) -> None:
        self.hardware = hardware
        self.timeout_s = timeout_s
        self.queue: Deque[str] = deque()
        self.last_spoken: Optional[str] = None
        self.cancelled = False
        self.failures = 0
        self._worker: Optional[threading.Thread] = None
        self._worker_started_t = 0.0
        self._worker_error: Optional[BaseException] = None

    def enqueue(self, text: str) -> None:
        text = text.strip()
        if not text or text == self.last_spoken:
            return
        self.queue.append(text)

    def cancel(self) -> None:
        self.queue.clear()
        self.cancelled = True

    def step(self) -> None:
        if self._worker is not None:
            if self._worker.is_alive():
                if time.monotonic() - self._worker_started_t > self.timeout_s:
                    self.failures += 1
                    self._worker = None
                return
            if self._worker_error is not None:
                self.failures += 1
                self._worker_error = None
            self._worker = None

        if not self.queue:
            return
        text = self.queue.popleft()
        self._worker_started_t = time.monotonic()
        self._worker = threading.Thread(target=self._speak_worker, args=(text,), name="museum-speech", daemon=True)
        self._worker.start()

    def _speak_worker(self, text: str) -> None:
        try:
            self.hardware.speak(text)
            self.last_spoken = text
            self.cancelled = False
        except (HardwareCapabilityError, HardwareFault) as exc:
            self._worker_error = exc

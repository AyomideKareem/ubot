import json
import threading
import time
from typing import Dict, Optional

from .config import MuseumGuideConfig
from .models import AIResult, PerceptionFrame


class VisionAIProvider:
    def identify(self, frame: PerceptionFrame) -> AIResult:
        raise NotImplementedError


class FakeVisionAIProvider(VisionAIProvider):
    def __init__(self, result: Optional[AIResult] = None, fail: bool = False, delay_s: float = 0.0) -> None:
        self.result = result or AIResult(
            candidate_name="unverified display object",
            category="museum display",
            confidence=0.72,
            visible_evidence=["object appears inside a display area"],
            short_description="This appears to be a museum display object, but the exact identity is uncertain.",
            uncertainty="No catalogue match was available.",
            needs_human_review=True,
            safety_or_privacy_flags=[],
        )
        self.fail = fail
        self.delay_s = delay_s
        self.calls = 0

    def identify(self, frame: PerceptionFrame) -> AIResult:
        self.calls += 1
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("fake AI failure")
        if frame.people():
            return AIResult(
                candidate_name="person",
                category="person",
                confidence=1.0,
                visible_evidence=["person detected in frame"],
                short_description="I will not identify visitors as museum artifacts.",
                uncertainty="The target region contains a person.",
                needs_human_review=False,
                safety_or_privacy_flags=["person_detected"],
            )
        self.result.validate()
        return self.result


class CachedVisionAIProvider(VisionAIProvider):
    def __init__(self, provider: VisionAIProvider) -> None:
        self.provider = provider
        self.cache: Dict[str, AIResult] = {}

    def identify(self, frame: PerceptionFrame) -> AIResult:
        key = _frame_key(frame)
        if key in self.cache:
            return self.cache[key]
        result = self.provider.identify(frame)
        result.validate()
        self.cache[key] = result
        return result


def result_from_json(payload: str) -> AIResult:
    data = json.loads(payload)
    if not isinstance(data.get("visible_evidence"), list) or not all(
        isinstance(item, str) for item in data["visible_evidence"]
    ):
        raise ValueError("visible_evidence must be a list of strings")
    if not isinstance(data.get("safety_or_privacy_flags"), list) or not all(
        isinstance(item, str) for item in data["safety_or_privacy_flags"]
    ):
        raise ValueError("safety_or_privacy_flags must be a list of strings")
    if not isinstance(data.get("needs_human_review"), bool):
        raise ValueError("needs_human_review must be a bool")
    result = AIResult(
        candidate_name=data["candidate_name"],
        category=data["category"],
        confidence=float(data["confidence"]),
        visible_evidence=data["visible_evidence"],
        short_description=data["short_description"],
        uncertainty=data["uncertainty"],
        needs_human_review=data["needs_human_review"],
        safety_or_privacy_flags=data["safety_or_privacy_flags"],
    )
    result.validate()
    return result


def _frame_key(frame: PerceptionFrame) -> str:
    labels = [
        "%s:%s:%0.2f" % (d.kind.value, d.track_id or d.label, d.confidence)
        for d in frame.detections
    ]
    return "|".join(labels) or "empty"


class VisionAIJob:
    def __init__(self, provider: VisionAIProvider, frame: PerceptionFrame) -> None:
        self.provider = provider
        self.frame = frame
        self.result: Optional[AIResult] = None
        self.error: Optional[BaseException] = None
        self.started_t = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="museum-vision-ai", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self.provider.identify(self.frame)
            self.result.validate()
        except BaseException as exc:
            self.error = exc

    def done(self) -> bool:
        return not self._thread.is_alive()

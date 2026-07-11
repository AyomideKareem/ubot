from dataclasses import dataclass
from typing import Dict, Optional

from .config import MuseumGuideConfig
from .models import DetectionKind, PerceptionFrame, VisionDetection


@dataclass
class ArtifactTrack:
    key: str
    detection: VisionDetection
    count: int = 1
    last_timestamp: float = 0.0


class ArtifactCandidateTracker:
    def __init__(self, cfg: MuseumGuideConfig) -> None:
        self.cfg = cfg
        self._tracks: Dict[str, ArtifactTrack] = {}

    def update(self, frame: PerceptionFrame) -> Optional[ArtifactTrack]:
        if not frame.valid:
            self._tracks.clear()
            return None
        if frame.people():
            # Do not consider any candidate when people are in view; privacy and safety first.
            self._tracks.clear()
            return None

        best = self._best_artifact(frame)
        if best is None:
            self._tracks.clear()
            return None

        key = best.track_id or _region_key(best)
        track = self._tracks.get(key)
        if track is None:
            track = ArtifactTrack(key=key, detection=best, count=1, last_timestamp=frame.timestamp)
        else:
            track.detection = best
            track.count += 1
            track.last_timestamp = frame.timestamp
        self._tracks = {key: track}
        if track.count >= self.cfg.confirmation_frames:
            return track
        return None

    def reset(self) -> None:
        self._tracks.clear()

    def _best_artifact(self, frame: PerceptionFrame) -> Optional[VisionDetection]:
        candidates = [
            detection
            for detection in frame.detections
            if detection.kind == DetectionKind.ARTIFACT
            and detection.confidence >= self.cfg.artifact_confidence_threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda detection: detection.confidence)


def _region_key(detection: VisionDetection) -> str:
    x, y, w, h = detection.bbox
    return "%d:%d:%d:%d" % (round(x * 10), round(y * 10), round(w * 10), round(h * 10))


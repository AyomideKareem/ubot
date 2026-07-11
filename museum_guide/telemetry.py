import csv
import json
import time
from typing import Any, Dict, Optional


class StructuredLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._file = open(path, "a") if path else None
        self.events = []

    def event(self, name: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"timestamp": time.time(), "event": name}
        payload.update(fields)
        self.events.append(payload)
        line = json.dumps(payload, sort_keys=True)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


class CsvTelemetry:
    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestamp",
                "state",
                "distance_center_m",
                "distance_left_m",
                "distance_right_m",
                "safety_state",
                "motion_kind",
                "motion_speed",
                "reason",
            ]
        )

    def row(
        self,
        state: str,
        distance_center_m: Any,
        distance_left_m: Any,
        distance_right_m: Any,
        safety_state: str,
        motion_kind: str,
        motion_speed: float,
        reason: str,
    ) -> None:
        self._writer.writerow(
            [
                time.time(),
                state,
                distance_center_m,
                distance_left_m,
                distance_right_m,
                safety_state,
                motion_kind,
                motion_speed,
                reason,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


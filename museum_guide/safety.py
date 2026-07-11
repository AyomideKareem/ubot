import time
from typing import Optional

from .config import MuseumGuideConfig
from .models import DistanceReading, PerceptionFrame, SafetyState


class SafetySupervisor:
    def __init__(self, cfg: MuseumGuideConfig) -> None:
        self.cfg = cfg
        self.emergency_stop_requested = False
        self.last_motion_command_t = 0.0
        self.last_fault: Optional[str] = None

    def request_emergency_stop(self, reason: str = "operator") -> None:
        self.emergency_stop_requested = True
        self.last_fault = reason

    def mark_motion_command(self, now: Optional[float] = None) -> None:
        self.last_motion_command_t = time.monotonic() if now is None else now

    def evaluate(
        self,
        distance: Optional[DistanceReading],
        frame: Optional[PerceptionFrame],
        balance_active: bool,
        tilt_deg: Optional[float] = None,
        now: Optional[float] = None,
    ) -> SafetyState:
        now = time.monotonic() if now is None else now
        if self.emergency_stop_requested:
            return SafetyState.EMERGENCY_STOP
        if not balance_active:
            self.last_fault = "balance inactive"
            return SafetyState.FAULT
        if tilt_deg is not None and abs(tilt_deg) > self.cfg.excessive_tilt_deg:
            self.last_fault = "excessive tilt"
            return SafetyState.FAULT
        if distance is None or not distance.valid or now - distance.timestamp > self.cfg.sensor_timeout_s:
            self.last_fault = "invalid or stale distance"
            return SafetyState.DANGER
        if frame is None or not frame.valid:
            self.last_fault = "invalid camera frame"
            return SafetyState.CAUTION
        if frame.timestamp > 0.0 and now - frame.timestamp > self.cfg.camera_timeout_s:
            self.last_fault = "stale camera frame"
            return SafetyState.CAUTION
        if frame is not None and frame.people():
            person_distance = min(
                [
                    d.distance_est_m
                    for d in frame.people()
                    if d.distance_est_m is not None
                ]
                or [distance.center_m if distance.center_m is not None else 999.0]
            )
            if person_distance < self.cfg.human_safe_distance_m:
                return SafetyState.DANGER
            return SafetyState.CAUTION
        if distance.center_m is None:
            return SafetyState.DANGER
        if distance.center_m <= self.cfg.danger_distance_m:
            return SafetyState.DANGER
        if distance.center_m <= self.cfg.caution_distance_m:
            return SafetyState.CAUTION
        return SafetyState.OK

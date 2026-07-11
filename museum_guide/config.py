from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MuseumGuideConfig:
    robot_ip: str = "192.168.1.77"
    patrol_speed_cm_s: float = 6.0
    caution_speed_cm_s: float = 3.0
    reverse_speed_cm_s: float = 5.0
    turn_speed: float = 4.0
    acceleration_limit_cm_s2: float = 8.0
    clear_distance_m: float = 1.20
    caution_distance_m: float = 0.80
    danger_distance_m: float = 0.45
    human_safe_distance_m: float = 1.50
    desired_artifact_distance_m: float = 3.0
    distance_tolerance_m: float = 0.25
    sensor_timeout_s: float = 0.50
    camera_timeout_s: float = 1.00
    confirmation_frames: int = 3
    ai_confidence_threshold: float = 0.55
    ai_timeout_s: float = 5.0
    speech_timeout_s: float = 5.0
    artifact_confidence_threshold: float = 0.60
    capture_sharpness_threshold: float = 0.45
    capture_brightness_threshold: float = 0.20
    max_capture_attempts: int = 5
    max_recovery_attempts: int = 3
    max_reverse_duration_s: float = 1.2
    max_turn_duration_s: float = 1.0
    movement_command_timeout_s: float = 0.40
    excessive_tilt_deg: float = 14.0
    low_motion_hold_speed_cm_s: float = 0.0
    telemetry_csv_path: str = ""
    log_level: str = "INFO"
    retain_images: bool = False
    privacy_blur_people: bool = True
    allow_physical_movement: bool = False
    state_timeouts_s: Dict[str, float] = field(
        default_factory=lambda: {
            "STARTUP": 3.0,
            "CALIBRATING": 30.0,
            "PATROLLING": 0.0,
            "ARTIFACT_CANDIDATE": 5.0,
            "APPROACHING": 12.0,
            "POSITIONING": 5.0,
            "CAPTURING": 8.0,
            "IDENTIFYING": 15.0,
            "PRESENTING": 10.0,
            "BACKING_AWAY": 2.0,
            "TURNING": 2.0,
            "RECOVERING": 8.0,
            "SAFE_SHUTDOWN": 3.0,
            "FAULT": 0.0,
        }
    )

    def validate(self) -> None:
        if not (self.danger_distance_m < self.caution_distance_m < self.clear_distance_m):
            raise ValueError("Distance zones must satisfy danger < caution < clear")
        if self.human_safe_distance_m < self.caution_distance_m:
            raise ValueError("Human-safe distance should be at least the caution zone")
        if self.desired_artifact_distance_m <= 0:
            raise ValueError("desired_artifact_distance_m must be positive")
        if self.confirmation_frames < 2:
            raise ValueError("confirmation_frames must require multiple observations")
        if self.patrol_speed_cm_s <= 0:
            raise ValueError("patrol speed must be positive")
        if self.reverse_speed_cm_s <= 0:
            raise ValueError("reverse speed must be positive")

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GuideState(str, Enum):
    STARTUP = "STARTUP"
    CALIBRATING = "CALIBRATING"
    PATROLLING = "PATROLLING"
    ARTIFACT_CANDIDATE = "ARTIFACT_CANDIDATE"
    APPROACHING = "APPROACHING"
    POSITIONING = "POSITIONING"
    CAPTURING = "CAPTURING"
    IDENTIFYING = "IDENTIFYING"
    PRESENTING = "PRESENTING"
    BACKING_AWAY = "BACKING_AWAY"
    TURNING = "TURNING"
    RECOVERING = "RECOVERING"
    SAFE_SHUTDOWN = "SAFE_SHUTDOWN"
    FAULT = "FAULT"


class SafetyState(str, Enum):
    OK = "OK"
    CAUTION = "CAUTION"
    DANGER = "DANGER"
    FAULT = "FAULT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class DetectionKind(str, Enum):
    ARTIFACT = "artifact"
    PERSON = "person"
    WALL = "wall"
    DISPLAY_CASE = "display_case"
    FREE_SPACE = "free_space"
    UNKNOWN = "unknown"


class MotionKind(str, Enum):
    HOLD = "hold"
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass
class DistanceReading:
    center_m: Optional[float]
    left_m: Optional[float] = None
    right_m: Optional[float] = None
    timestamp: float = 0.0
    valid: bool = True
    source: str = "unknown"


@dataclass
class VisionDetection:
    kind: DetectionKind
    confidence: float
    bbox: Tuple[float, float, float, float]
    label: str = ""
    track_id: Optional[str] = None
    distance_est_m: Optional[float] = None

    def center_x(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0


@dataclass
class PerceptionFrame:
    timestamp: float
    detections: List[VisionDetection] = field(default_factory=list)
    sharpness: float = 1.0
    brightness: float = 1.0
    valid: bool = True
    image: Any = None

    def people(self) -> List[VisionDetection]:
        return [d for d in self.detections if d.kind == DetectionKind.PERSON]

    def artifacts(self) -> List[VisionDetection]:
        return [d for d in self.detections if d.kind == DetectionKind.ARTIFACT]


@dataclass
class MovementCommand:
    kind: MotionKind
    speed: float = 0.0
    duration_s: Optional[float] = None
    reason: str = ""


@dataclass
class AIResult:
    candidate_name: str
    category: str
    confidence: float
    visible_evidence: List[str]
    short_description: str
    uncertainty: str
    needs_human_review: bool
    safety_or_privacy_flags: List[str]

    @classmethod
    def uncertain(cls, reason: str) -> "AIResult":
        return cls(
            candidate_name="unknown",
            category="unknown",
            confidence=0.0,
            visible_evidence=[],
            short_description="I cannot identify this display confidently from the current view.",
            uncertainty=reason,
            needs_human_review=True,
            safety_or_privacy_flags=[],
        )

    def validate(self) -> None:
        required_text = (
            self.candidate_name,
            self.category,
            self.short_description,
            self.uncertainty,
        )
        if any(not isinstance(value, str) for value in required_text):
            raise ValueError("AI result text fields must be strings")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("AI confidence must be 0..1")
        if not isinstance(self.visible_evidence, list):
            raise ValueError("visible_evidence must be a list")
        if not isinstance(self.safety_or_privacy_flags, list):
            raise ValueError("safety_or_privacy_flags must be a list")


@dataclass
class StateTransition:
    timestamp: float
    from_state: GuideState
    to_state: GuideState
    reason: str


@dataclass
class StepResult:
    state: GuideState
    command: MovementCommand
    safety_state: SafetyState
    reason: str = ""
    ai_result: Optional[AIResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


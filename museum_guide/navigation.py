import time
from typing import Optional

from .ai import VisionAIJob, VisionAIProvider
from .artifacts import ArtifactCandidateTracker, ArtifactTrack
from .config import MuseumGuideConfig
from .hardware import HardwareCapabilityError, HardwareFault, MuseumHardware
from .models import (
    AIResult,
    DistanceReading,
    GuideState,
    MotionKind,
    MovementCommand,
    PerceptionFrame,
    SafetyState,
    StateTransition,
    StepResult,
)
from .safety import SafetySupervisor
from .speech import SpeechQueue
from .telemetry import CsvTelemetry, StructuredLogger


class MuseumGuideController:
    def __init__(
        self,
        cfg: MuseumGuideConfig,
        hardware: MuseumHardware,
        ai_provider: VisionAIProvider,
        logger: Optional[StructuredLogger] = None,
        telemetry: Optional[CsvTelemetry] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.hardware = hardware
        self.ai_provider = ai_provider
        self.logger = logger or StructuredLogger()
        self.telemetry = telemetry
        self.safety = SafetySupervisor(cfg)
        self.speech = SpeechQueue(hardware, timeout_s=cfg.speech_timeout_s)
        self.tracker = ArtifactCandidateTracker(cfg)
        self.state = GuideState.STARTUP
        self.state_enter_t = time.monotonic()
        self.transitions = []
        self.recovery_attempts = 0
        self.active_track: Optional[ArtifactTrack] = None
        self.capture_frame: Optional[PerceptionFrame] = None
        self.capture_candidates = []
        self.ai_result: Optional[AIResult] = None
        self.ai_job: Optional[VisionAIJob] = None
        self.turn_right_next = True
        self.last_command_speed = 0.0
        self.last_command_t = time.monotonic()

    def startup(self) -> None:
        try:
            self.hardware.start_balance()
            self.transition(GuideState.CALIBRATING, "balance controller active")
        except Exception as exc:
            self.transition(GuideState.FAULT, "balance startup failed: %s" % exc)

    def step(self, now: Optional[float] = None) -> StepResult:
        now = time.monotonic() if now is None else now
        if self.state == GuideState.STARTUP:
            self.startup()

        distance = self._safe_read_distance(now)
        frame = self._safe_read_frame(now)
        tilt_deg = self._safe_read_tilt()
        safety_state = self.safety.evaluate(distance, frame, self.hardware.balance_active(), tilt_deg=tilt_deg, now=now)

        if safety_state == SafetyState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
            self.transition(GuideState.SAFE_SHUTDOWN, "operator emergency stop")
            return StepResult(self.state, _hold("operator emergency stop"), safety_state, "operator emergency stop")
        if safety_state == SafetyState.FAULT:
            self.transition(GuideState.FAULT, self.safety.last_fault or "safety fault")
            return self._apply(self.state, _hold(self.safety.last_fault or "safety fault"), safety_state, now, distance)
        if safety_state == SafetyState.DANGER and self.state not in (
            GuideState.BACKING_AWAY,
            GuideState.TURNING,
            GuideState.RECOVERING,
            GuideState.FAULT,
        ):
            if distance.valid and distance.center_m is not None:
                self.transition(GuideState.BACKING_AWAY, "danger zone or unsafe person")
            else:
                self.transition(GuideState.RECOVERING, "unsafe or missing distance; avoid blind reverse")

        result = self._step_state(distance, frame, safety_state, now)
        self.speech.step()
        return result

    def transition(self, to_state: GuideState, reason: str) -> None:
        if to_state == self.state:
            return
        from_state = self.state
        self.state = to_state
        self.state_enter_t = time.monotonic()
        transition = StateTransition(self.state_enter_t, from_state, to_state, reason)
        self.transitions.append(transition)
        self.logger.event("state_transition", from_state=from_state.value, to_state=to_state.value, reason=reason)

    def _step_state(
        self,
        distance: DistanceReading,
        frame: PerceptionFrame,
        safety_state: SafetyState,
        now: float,
    ) -> StepResult:
        if self._timed_out(now):
            self.recovery_attempts += 1
            if self.recovery_attempts > self.cfg.max_recovery_attempts:
                self.transition(GuideState.FAULT, "repeated state timeouts")
                return self._apply(self.state, _hold("repeated timeout fault"), SafetyState.FAULT, now, distance)
            self.transition(GuideState.RECOVERING, "state timeout")

        if self.state == GuideState.CALIBRATING:
            self.transition(GuideState.PATROLLING, "built-in balance diagnostic must be passed before running")
            return self._apply(self.state, _hold("calibration gate complete"), safety_state, now, distance)

        if self.state == GuideState.PATROLLING:
            return self._patrol(distance, frame, safety_state, now)

        if self.state == GuideState.ARTIFACT_CANDIDATE:
            return self._artifact_candidate(distance, frame, safety_state, now)

        if self.state == GuideState.APPROACHING:
            return self._approach(distance, frame, safety_state, now)

        if self.state == GuideState.POSITIONING:
            self.transition(GuideState.CAPTURING, "target positioned")
            return self._apply(self.state, _hold("stabilize before capture"), safety_state, now, distance)

        if self.state == GuideState.CAPTURING:
            return self._capture(frame, safety_state, now, distance)

        if self.state == GuideState.IDENTIFYING:
            return self._identify(safety_state, now, distance)

        if self.state == GuideState.PRESENTING:
            return self._present(safety_state, now, distance)

        if self.state == GuideState.BACKING_AWAY:
            return self._back_away(distance, safety_state, now)

        if self.state == GuideState.TURNING:
            return self._turn(distance, safety_state, now)

        if self.state == GuideState.RECOVERING:
            if _is_clear(distance, self.cfg):
                self.transition(GuideState.PATROLLING, "clear after recovery")
            elif distance.valid and distance.center_m is not None:
                self.transition(GuideState.BACKING_AWAY, "recover by backing away")
            else:
                return self._apply(self.state, _hold("waiting for valid distance before reverse"), SafetyState.DANGER, now, distance)
            return self._apply(self.state, _hold("recovering"), safety_state, now, distance)

        if self.state in (GuideState.SAFE_SHUTDOWN, GuideState.FAULT):
            return self._apply(self.state, _hold("safe terminal state"), safety_state, now, distance)

        self.transition(GuideState.FAULT, "unknown state")
        return self._apply(self.state, _hold("unknown state"), SafetyState.FAULT, now, distance)

    def _patrol(
        self,
        distance: DistanceReading,
        frame: PerceptionFrame,
        safety_state: SafetyState,
        now: float,
    ) -> StepResult:
        if safety_state == SafetyState.CAUTION:
            speed = self.cfg.caution_speed_cm_s
        else:
            speed = self.cfg.patrol_speed_cm_s
        track = self.tracker.update(frame)
        if track is not None:
            self.active_track = track
            self.transition(GuideState.ARTIFACT_CANDIDATE, "artifact confirmed by multiple camera frames")
            return self._apply(self.state, _hold("confirmed artifact candidate"), safety_state, now, distance)
        return self._apply(
            self.state,
            MovementCommand(MotionKind.FORWARD, speed, reason="clear patrol" if safety_state == SafetyState.OK else "caution patrol"),
            safety_state,
            now,
            distance,
        )

    def _artifact_candidate(
        self,
        distance: DistanceReading,
        frame: PerceptionFrame,
        safety_state: SafetyState,
        now: float,
    ) -> StepResult:
        if frame.people():
            self.tracker.reset()
            self.transition(GuideState.BACKING_AWAY, "person detected near candidate")
            return self._apply(self.state, _reverse(self.cfg, "yield to person"), SafetyState.DANGER, now, distance)
        if not self.active_track:
            self.transition(GuideState.PATROLLING, "candidate missing")
            return self._apply(self.state, _hold("candidate missing"), safety_state, now, distance)
        self.transition(GuideState.APPROACHING, "safe candidate selected")
        return self._apply(self.state, _hold("prepare approach"), safety_state, now, distance)

    def _approach(
        self,
        distance: DistanceReading,
        frame: PerceptionFrame,
        safety_state: SafetyState,
        now: float,
    ) -> StepResult:
        if not self.active_track:
            self.transition(GuideState.RECOVERING, "target lost during approach")
            return self._apply(self.state, _hold("target lost"), SafetyState.DANGER, now, distance)
        if frame.people():
            self.transition(GuideState.BACKING_AWAY, "person crossed approach path")
            return self._apply(self.state, _reverse(self.cfg, "person in approach path"), SafetyState.DANGER, now, distance)
        target_distance = _target_distance(distance, self.active_track)
        if target_distance is None:
            self.transition(GuideState.RECOVERING, "target distance unavailable")
            return self._apply(self.state, _hold("no reliable target range"), SafetyState.DANGER, now, distance)
        if target_distance <= self.cfg.desired_artifact_distance_m + self.cfg.distance_tolerance_m:
            self.transition(GuideState.POSITIONING, "viewing distance reached")
            return self._apply(self.state, _hold("at viewing distance"), safety_state, now, distance)
        return self._apply(
            self.state,
            MovementCommand(MotionKind.FORWARD, self.cfg.caution_speed_cm_s, reason="slow artifact approach"),
            safety_state,
            now,
            distance,
        )

    def _capture(
        self,
        frame: PerceptionFrame,
        safety_state: SafetyState,
        now: float,
        distance: DistanceReading,
    ) -> StepResult:
        if not frame.valid or frame.people():
            self.capture_candidates = []
            self.transition(GuideState.BACKING_AWAY, "invalid/person-dominated capture")
            return self._apply(self.state, _reverse(self.cfg, "bad capture frame"), SafetyState.DANGER, now, distance)
        if frame.brightness >= self.cfg.capture_brightness_threshold:
            self.capture_candidates.append(frame)
        if len(self.capture_candidates) < self.cfg.max_capture_attempts:
            return self._apply(self.state, _hold("collecting capture frames"), safety_state, now, distance)

        best = max(self.capture_candidates, key=lambda candidate: candidate.sharpness)
        self.capture_candidates = []
        if best.sharpness < self.cfg.capture_sharpness_threshold:
            self.transition(GuideState.RECOVERING, "all capture frames too blurred")
            return self._apply(self.state, _hold("blurred capture rejected"), SafetyState.CAUTION, now, distance)
        self.capture_frame = best
        self.transition(GuideState.IDENTIFYING, "captured sharpest valid image")
        return self._apply(self.state, _hold("image captured"), safety_state, now, distance)

    def _identify(self, safety_state: SafetyState, now: float, distance: DistanceReading) -> StepResult:
        if self.capture_frame is None:
            self.transition(GuideState.RECOVERING, "no capture frame")
            return self._apply(self.state, _hold("missing capture"), SafetyState.DANGER, now, distance)
        try:
            if self.ai_job is None:
                self.ai_job = VisionAIJob(self.ai_provider, self.capture_frame)
                return self._apply(self.state, _hold("AI identification pending"), safety_state, now, distance)
            if now - self.ai_job.started_t > self.cfg.ai_timeout_s:
                self.ai_job = None
                result = AIResult.uncertain("AI identification timed out")
                self.ai_result = result
                self.transition(GuideState.PRESENTING, "AI timed out")
                return self._apply(self.state, _hold("AI timeout"), safety_state, now, distance, result)
            if not self.ai_job.done():
                return self._apply(self.state, _hold("AI identification pending"), safety_state, now, distance)
            if self.ai_job.error is not None:
                result = AIResult.uncertain("AI identification failed: %s" % self.ai_job.error)
            else:
                result = self.ai_job.result or AIResult.uncertain("AI returned no result")
            self.ai_job = None
        except Exception as exc:
            self.ai_job = None
            result = AIResult.uncertain("AI identification failed: %s" % exc)
        if result.category.lower() == "person" or "person_detected" in result.safety_or_privacy_flags:
            self.transition(GuideState.BACKING_AWAY, "AI/person privacy guard")
            return self._apply(self.state, _reverse(self.cfg, "AI rejected person"), SafetyState.DANGER, now, distance, result)
        self.ai_result = result
        self.transition(GuideState.PRESENTING, "AI result ready")
        return self._apply(self.state, _hold("AI result ready"), safety_state, now, distance, result)

    def _present(self, safety_state: SafetyState, now: float, distance: DistanceReading) -> StepResult:
        if self.ai_result and self.ai_result.confidence >= self.cfg.ai_confidence_threshold:
            self.speech.enqueue(self.ai_result.short_description)
            self.logger.event("spoken_description_queued", text=self.ai_result.short_description)
        self.transition(GuideState.BACKING_AWAY, "presentation complete")
        return self._apply(self.state, _hold("presenting description"), safety_state, now, distance, self.ai_result)

    def _back_away(self, distance: DistanceReading, safety_state: SafetyState, now: float) -> StepResult:
        if not distance.valid or distance.center_m is None:
            self.transition(GuideState.RECOVERING, "lost distance during back-away")
            return self._apply(self.state, _hold("no reverse without valid distance"), SafetyState.DANGER, now, distance)
        elapsed = now - self.state_enter_t
        if elapsed >= self.cfg.max_reverse_duration_s or _is_clear(distance, self.cfg):
            self.transition(GuideState.TURNING, "back-away complete")
            return self._apply(self.state, _hold("choose turn direction"), safety_state, now, distance)
        return self._apply(self.state, _reverse(self.cfg, "bounded back-away"), safety_state, now, distance)

    def _turn(self, distance: DistanceReading, safety_state: SafetyState, now: float) -> StepResult:
        elapsed = now - self.state_enter_t
        if elapsed >= self.cfg.max_turn_duration_s:
            if _is_clear(distance, self.cfg):
                self.transition(GuideState.PATROLLING, "clearance rechecked after turn")
                self.recovery_attempts = 0
            else:
                self.recovery_attempts += 1
                self.transition(GuideState.RECOVERING, "still blocked after turn")
            return self._apply(self.state, _hold("turn complete"), safety_state, now, distance)
        command = self._turn_command(distance)
        return self._apply(self.state, command, safety_state, now, distance)

    def _turn_command(self, distance: DistanceReading) -> MovementCommand:
        if distance.left_m is not None and distance.right_m is not None:
            if distance.left_m > distance.right_m:
                return MovementCommand(MotionKind.TURN_LEFT, self.cfg.turn_speed, reason="turn toward greater left clearance")
            return MovementCommand(MotionKind.TURN_RIGHT, self.cfg.turn_speed, reason="turn toward greater right clearance")
        self.turn_right_next = not self.turn_right_next
        kind = MotionKind.TURN_RIGHT if self.turn_right_next else MotionKind.TURN_LEFT
        return MovementCommand(kind, self.cfg.turn_speed, reason="bounded alternating turn; directional range unavailable")

    def _safe_read_distance(self, now: float) -> DistanceReading:
        try:
            return self.hardware.read_distance()
        except (HardwareCapabilityError, HardwareFault) as exc:
            self.logger.event("distance_fault", error=str(exc))
            return DistanceReading(center_m=None, timestamp=now, valid=False, source="fault")

    def _safe_read_frame(self, now: float) -> PerceptionFrame:
        try:
            return self.hardware.read_camera_frame()
        except (HardwareCapabilityError, HardwareFault) as exc:
            self.logger.event("camera_fault", error=str(exc))
            return PerceptionFrame(timestamp=now, valid=False)

    def _safe_read_tilt(self) -> Optional[float]:
        try:
            return self.hardware.read_tilt_deg()
        except (HardwareCapabilityError, HardwareFault) as exc:
            self.logger.event("tilt_fault", error=str(exc))
            return None

    def _apply(
        self,
        state: GuideState,
        command: MovementCommand,
        safety_state: SafetyState,
        now: float,
        distance: Optional[DistanceReading] = None,
        ai_result: Optional[AIResult] = None,
    ) -> StepResult:
        try:
            command = self._limit_acceleration(command, now)
            if state in (GuideState.FAULT, GuideState.SAFE_SHUTDOWN):
                self.hardware.stop_translation_keep_balance()
            else:
                self.hardware.command_motion(command)
            self.safety.mark_motion_command(now)
        except Exception as exc:
            self.logger.event("motion_fault", error=str(exc), state=state.value)
            self.transition(GuideState.FAULT, "motion command failed")
            try:
                self.hardware.stop_translation_keep_balance()
            except Exception as stop_exc:
                self.logger.event("motion_stop_fault", error=str(stop_exc))
            command = _hold("motion fault")
            safety_state = SafetyState.FAULT

        if distance is not None and self.telemetry is not None:
            self.telemetry.row(
                self.state.value,
                distance.center_m,
                distance.left_m,
                distance.right_m,
                safety_state.value,
                command.kind.value,
                command.speed,
                command.reason,
            )
        self.logger.event(
            "step",
            state=self.state.value,
            safety_state=safety_state.value,
            motion_kind=command.kind.value,
            speed=command.speed,
            reason=command.reason,
        )
        return StepResult(self.state, command, safety_state, command.reason, ai_result)

    def _limit_acceleration(self, command: MovementCommand, now: float) -> MovementCommand:
        if command.kind == MotionKind.HOLD:
            self.last_command_speed = 0.0
            self.last_command_t = now
            return command
        dt = max(0.0, now - self.last_command_t)
        max_delta = self.cfg.acceleration_limit_cm_s2 * dt
        limited_speed = command.speed
        if max_delta > 0.0:
            limited_speed = max(
                self.last_command_speed - max_delta,
                min(self.last_command_speed + max_delta, command.speed),
            )
        self.last_command_speed = limited_speed
        self.last_command_t = now
        if limited_speed == command.speed:
            return command
        return MovementCommand(command.kind, limited_speed, command.duration_s, command.reason + " (accel limited)")

    def _timed_out(self, now: float) -> bool:
        timeout = self.cfg.state_timeouts_s.get(self.state.value, 0.0)
        return timeout > 0.0 and now - self.state_enter_t > timeout


def _hold(reason: str) -> MovementCommand:
    return MovementCommand(MotionKind.HOLD, 0.0, reason=reason)


def _reverse(cfg: MuseumGuideConfig, reason: str) -> MovementCommand:
    return MovementCommand(MotionKind.BACKWARD, cfg.reverse_speed_cm_s, duration_s=cfg.max_reverse_duration_s, reason=reason)


def _is_clear(distance: DistanceReading, cfg: MuseumGuideConfig) -> bool:
    return distance.valid and distance.center_m is not None and distance.center_m >= cfg.clear_distance_m


def _target_distance(distance: DistanceReading, track: Optional[ArtifactTrack]) -> Optional[float]:
    if distance.valid:
        return distance.center_m
    if track is not None and track.detection.distance_est_m is not None:
        return track.detection.distance_est_m
    return None

import time
import unittest

from museum_guide.ai import FakeVisionAIProvider
from museum_guide.ai import result_from_json
from museum_guide.config import MuseumGuideConfig
from museum_guide.hardware import FakeMuseumHardware
from museum_guide.hardware import HardwareCapabilityError
from museum_guide.hardware import UGOTMuseumHardware
from museum_guide.models import (
    AIResult,
    DetectionKind,
    DistanceReading,
    GuideState,
    MotionKind,
    MovementCommand,
    PerceptionFrame,
    SafetyState,
    VisionDetection,
)
from museum_guide.navigation import MuseumGuideController
from museum_guide.artifacts import ArtifactTrack


def cfg() -> MuseumGuideConfig:
    config = MuseumGuideConfig()
    config.state_timeouts_s["CALIBRATING"] = 2.0
    return config


def controller_with(fake: FakeMuseumHardware, ai=None) -> MuseumGuideController:
    return MuseumGuideController(cfg(), fake, ai or FakeVisionAIProvider())


def frame_with(*detections: VisionDetection, sharpness=1.0, brightness=1.0) -> PerceptionFrame:
    return PerceptionFrame(
        timestamp=time.monotonic(),
        detections=list(detections),
        sharpness=sharpness,
        brightness=brightness,
        valid=True,
    )


def artifact(track_id="a1", confidence=0.8, distance=3.5) -> VisionDetection:
    return VisionDetection(
        kind=DetectionKind.ARTIFACT,
        confidence=confidence,
        bbox=(0.4, 0.25, 0.2, 0.4),
        label="display",
        track_id=track_id,
        distance_est_m=distance,
    )


def step_until(controller, expected_state, limit=20):
    result = None
    for _ in range(limit):
        result = controller.step()
        if result.state == expected_state:
            return result
        time.sleep(0.01)
    return result


def person(distance=1.0) -> VisionDetection:
    return VisionDetection(
        kind=DetectionKind.PERSON,
        confidence=0.9,
        bbox=(0.3, 0.2, 0.2, 0.5),
        label="person",
        distance_est_m=distance,
    )


class MuseumGuideTests(unittest.TestCase):
    def test_clear_path_patrol_keeps_balance_active(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        result = c.step()
        self.assertEqual(result.state, GuideState.PATROLLING)
        self.assertEqual(result.command.kind, MotionKind.FORWARD)
        self.assertTrue(fake.balance_active())
        self.assertFalse(fake.balance_stop_called)

    def test_noisy_invalid_distance_is_unsafe(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.distance_queue.append(DistanceReading(None, timestamp=time.monotonic(), valid=False))
        result = c.step()
        self.assertEqual(result.state, GuideState.RECOVERING)
        self.assertEqual(result.command.kind, MotionKind.HOLD)

    def test_stale_distance_is_unsafe_and_does_not_reverse_blindly(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.distance_queue.append(DistanceReading(0.3, timestamp=time.monotonic() - 99.0, valid=True))
        result = c.step()
        self.assertIn(result.state, (GuideState.RECOVERING, GuideState.BACKING_AWAY))
        self.assertNotEqual(result.command.kind, MotionKind.FORWARD)

    def test_wall_appearing_ahead_triggers_backaway(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.distance_queue.append(DistanceReading(0.30, timestamp=time.monotonic(), valid=True))
        result = c.step()
        self.assertEqual(result.state, GuideState.BACKING_AWAY)
        self.assertEqual(result.command.kind, MotionKind.BACKWARD)

    def test_turns_toward_clearer_side(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.transition(GuideState.TURNING, "test")
        fake.distance_queue.append(DistanceReading(0.6, left_m=1.8, right_m=0.7, timestamp=time.monotonic(), valid=True))
        result = c.step()
        self.assertEqual(result.command.kind, MotionKind.TURN_LEFT)

    def test_person_entering_path_is_not_artifact_and_causes_yield(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.frame_queue.append(frame_with(person(distance=0.9)))
        result = c.step()
        self.assertIn(result.state, (GuideState.BACKING_AWAY, GuideState.TURNING))
        self.assertNotEqual(result.command.kind, MotionKind.FORWARD)

    def test_artifact_candidate_requires_multiple_frames(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.frame_queue.extend([frame_with(artifact()), frame_with(artifact()), frame_with(artifact())])
        c.step()
        self.assertEqual(c.state, GuideState.PATROLLING)
        result = c.step()
        if result.state == GuideState.PATROLLING:
            result = c.step()
        self.assertEqual(result.state, GuideState.ARTIFACT_CANDIDATE)

    def test_false_artifact_rejected_below_confidence(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.frame_queue.extend([frame_with(artifact(confidence=0.2)) for _ in range(4)])
        for _ in range(4):
            result = c.step()
        self.assertEqual(result.state, GuideState.PATROLLING)

    def test_distance_only_never_creates_artifact_candidate(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        for _ in range(5):
            fake.distance_queue.append(DistanceReading(3.0, timestamp=time.monotonic(), valid=True))
            result = c.step()
        self.assertEqual(result.state, GuideState.PATROLLING)
        self.assertIsNone(c.active_track)

    def test_target_lost_during_approach_recovers(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.transition(GuideState.APPROACHING, "test")
        c.active_track = None
        result = c.step()
        self.assertEqual(result.state, GuideState.RECOVERING)

    def test_reaching_viewing_distance_positions_for_capture(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.transition(GuideState.APPROACHING, "test")
        c.active_track = ArtifactTrack("a1", artifact(distance=3.1), count=3, last_timestamp=time.monotonic())
        fake.distance_queue.append(DistanceReading(3.1, timestamp=time.monotonic(), valid=True))
        result = c.step()
        self.assertEqual(result.state, GuideState.POSITIONING)

    def test_image_blur_rejection_holds(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.transition(GuideState.CAPTURING, "test")
        fake.frame_queue.append(frame_with(artifact(), sharpness=0.1))
        result = c.step()
        self.assertEqual(result.state, GuideState.CAPTURING)
        self.assertEqual(result.command.kind, MotionKind.HOLD)

    def test_capture_selects_sharpest_valid_image(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.transition(GuideState.CAPTURING, "test")
        frames = [
            frame_with(artifact(), sharpness=0.5),
            frame_with(artifact(), sharpness=0.9),
            frame_with(artifact(), sharpness=0.7),
            frame_with(artifact(), sharpness=0.6),
            frame_with(artifact(), sharpness=0.8),
        ]
        fake.frame_queue.extend(frames)
        result = None
        for _ in frames:
            result = c.step()
        self.assertEqual(result.state, GuideState.IDENTIFYING)
        self.assertAlmostEqual(c.capture_frame.sharpness, 0.9)

    def test_ai_timeout_or_failure_becomes_uncertain_not_crash(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake, FakeVisionAIProvider(fail=True))
        c.step()
        c.capture_frame = frame_with(artifact())
        c.transition(GuideState.IDENTIFYING, "test")
        result = step_until(c, GuideState.PRESENTING)
        self.assertEqual(result.state, GuideState.PRESENTING)
        self.assertTrue(c.ai_result.needs_human_review)

    def test_ai_timeout_becomes_uncertain(self):
        fake = FakeMuseumHardware()
        config = cfg()
        config.ai_timeout_s = 0.01
        c = MuseumGuideController(config, fake, FakeVisionAIProvider(delay_s=1.0))
        c.step()
        c.capture_frame = frame_with(artifact())
        c.transition(GuideState.IDENTIFYING, "test")
        c.step()
        time.sleep(0.02)
        result = c.step()
        self.assertEqual(result.state, GuideState.PRESENTING)
        self.assertIn("timed out", c.ai_result.uncertainty)

    def test_invalid_ai_person_response_backs_away(self):
        fake = FakeMuseumHardware()
        ai = FakeVisionAIProvider(
            AIResult(
                candidate_name="person",
                category="person",
                confidence=1.0,
                visible_evidence=["face"],
                short_description="I will not identify visitors.",
                uncertainty="person detected",
                needs_human_review=False,
                safety_or_privacy_flags=["person_detected"],
            )
        )
        c = controller_with(fake, ai)
        c.step()
        c.capture_frame = frame_with(artifact())
        c.transition(GuideState.IDENTIFYING, "test")
        result = step_until(c, GuideState.BACKING_AWAY)
        self.assertEqual(result.state, GuideState.BACKING_AWAY)
        self.assertEqual(result.command.kind, MotionKind.BACKWARD)

    def test_speech_failure_does_not_disable_balance(self):
        fake = FakeMuseumHardware()
        fake.fail_speech = True
        c = controller_with(fake)
        c.step()
        c.ai_result = FakeVisionAIProvider().result
        c.transition(GuideState.PRESENTING, "test")
        c.step()
        c.speech.step()
        self.assertTrue(fake.balance_active())
        self.assertGreaterEqual(c.speech.failures, 1)

    def test_slow_speech_does_not_block_control_step(self):
        class SlowSpeechHardware(FakeMuseumHardware):
            def speak(self, text):
                time.sleep(0.2)
                super().speak(text)

        fake = SlowSpeechHardware()
        config = cfg()
        config.speech_timeout_s = 1.0
        c = MuseumGuideController(config, fake, FakeVisionAIProvider())
        c.step()
        c.speech.enqueue("short description")
        start = time.monotonic()
        c.speech.step()
        result = c.step()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.1)
        self.assertTrue(fake.balance_active())
        self.assertNotEqual(result.state, GuideState.FAULT)

    def test_communication_loss_enters_fault(self):
        fake = FakeMuseumHardware()
        fake.fail_motion = True
        c = controller_with(fake)
        result = c.step()
        self.assertEqual(result.safety_state, SafetyState.FAULT)
        self.assertEqual(c.state, GuideState.FAULT)

    def test_motion_fault_attempts_hold(self):
        class FaultyAfterRecording(FakeMuseumHardware):
            def __init__(self):
                super().__init__()
                self.stop_translation_calls = 0

            def command_motion(self, command):
                raise RuntimeError("movement failed")

            def stop_translation_keep_balance(self):
                self.stop_translation_calls += 1

        fake = FaultyAfterRecording()
        c = controller_with(fake)
        result = c.step()
        self.assertEqual(result.safety_state, SafetyState.FAULT)
        self.assertEqual(fake.stop_translation_calls, 1)

    def test_excessive_tilt_fault(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.tilt_deg = 30.0
        result = c.step()
        self.assertEqual(result.safety_state, SafetyState.FAULT)
        self.assertEqual(result.state, GuideState.FAULT)

    def test_repeated_recovery_failure_faults(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.recovery_attempts = c.cfg.max_recovery_attempts + 1
        c.transition(GuideState.RECOVERING, "test")
        fake.distance_queue.append(DistanceReading(0.2, timestamp=time.monotonic(), valid=True))
        result = c.step(now=c.state_enter_t + 99.0)
        self.assertEqual(result.state, GuideState.FAULT)

    def test_emergency_stop(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        c.safety.request_emergency_stop("test")
        result = c.step()
        self.assertEqual(result.state, GuideState.SAFE_SHUTDOWN)
        self.assertEqual(result.command.kind, MotionKind.HOLD)
        self.assertTrue(fake.emergency_stopped)

    def test_camera_failure_slows_or_recovers_not_clear_forward(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.fail_camera = True
        result = c.step()
        self.assertNotEqual(result.safety_state, SafetyState.OK)

    def test_stale_camera_frame_is_caution(self):
        fake = FakeMuseumHardware()
        c = controller_with(fake)
        c.step()
        fake.frame_queue.append(PerceptionFrame(timestamp=time.monotonic() - 99.0, valid=True))
        result = c.step()
        self.assertEqual(result.safety_state, SafetyState.CAUTION)

    def test_physical_adapter_blocks_movement_without_confirmation(self):
        class Robot:
            def balance_start_balancing(self):
                pass

            def balance_move_speed(self, direction, speed):
                pass

        config = cfg()
        config.allow_physical_movement = False
        adapter = UGOTMuseumHardware(config, robot=Robot())
        adapter._balance_active = True
        with self.assertRaises(HardwareCapabilityError):
            adapter.command_motion(MovementCommand(MotionKind.FORWARD, 5.0))

    def test_ai_json_schema_validation_rejects_invalid_confidence(self):
        payload = (
            '{"candidate_name":"x","category":"artifact","confidence":2.0,'
            '"visible_evidence":[],"short_description":"x","uncertainty":"high",'
            '"needs_human_review":true,"safety_or_privacy_flags":[]}'
        )
        with self.assertRaises(ValueError):
            result_from_json(payload)

    def test_ai_json_schema_rejects_string_flags(self):
        payload = (
            '{"candidate_name":"x","category":"artifact","confidence":0.5,'
            '"visible_evidence":"not-list","short_description":"x","uncertainty":"high",'
            '"needs_human_review":"yes","safety_or_privacy_flags":"person_detected"}'
        )
        with self.assertRaises(ValueError):
            result_from_json(payload)


if __name__ == "__main__":
    unittest.main()

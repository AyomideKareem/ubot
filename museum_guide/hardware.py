import time
from collections import deque
from typing import Any, Callable, Deque, Iterable, List, Optional

from .config import MuseumGuideConfig
from .models import DistanceReading, MotionKind, MovementCommand, PerceptionFrame


class HardwareCapabilityError(RuntimeError):
    pass


class HardwareFault(RuntimeError):
    pass


class MuseumHardware:
    def start_balance(self) -> None:
        raise NotImplementedError

    def keep_balance_active(self) -> None:
        raise NotImplementedError

    def command_motion(self, command: MovementCommand) -> None:
        raise NotImplementedError

    def read_distance(self) -> DistanceReading:
        raise NotImplementedError

    def read_camera_frame(self) -> PerceptionFrame:
        raise NotImplementedError

    def read_tilt_deg(self) -> Optional[float]:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        raise NotImplementedError

    def stop_translation_keep_balance(self) -> None:
        raise NotImplementedError

    def emergency_stop(self) -> None:
        raise NotImplementedError

    def balance_active(self) -> bool:
        raise NotImplementedError


class UGOTMuseumHardware(MuseumHardware):
    """Physical UGOT adapter.

    Confirmed repo APIs: initialize, DEVICE.getDeviceModel, balance_start_balancing,
    balance_move_speed, open_camera, read_camera_data. Distance, TTS, and balance
    turning are SDK-dependent and must be verified on the installed robot.
    """

    def __init__(self, cfg: MuseumGuideConfig, robot: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.robot = robot
        self._balance_active = False
        self._last_motion_t = 0.0
        self._distance_method: Optional[Callable[..., Any]] = None
        self._tts_method: Optional[Callable[..., Any]] = None
        self._turn_method: Optional[Callable[..., Any]] = None

    def connect(self) -> None:
        if self.robot is None:
            from ugot import ugot

            self.robot = ugot.UGOT()
            self.robot.initialize(self.cfg.robot_ip)
        self._distance_method = self._find_method(
            (
                "read_ultrasonic_distance",
                "get_ultrasonic_distance",
                "get_ultrasonic_data",
                "read_distance",
                "get_distance",
                "get_sonar_distance",
            )
        )
        self._tts_method = self._find_method(
            (
                "tts",
                "speak",
                "speak_text",
                "play_tts",
                "text_to_speech",
                "play_audio_tts",
            )
        )
        self._turn_method = self._find_method(
            (
                "balance_turn",
                "balance_turn_speed",
                "balance_rotate",
            )
        )
        try:
            self.robot.open_camera()
        except Exception as exc:
            raise HardwareFault(f"Camera open failed: {exc}")

    def get_chassis_mode(self) -> Optional[str]:
        try:
            return self.robot.DEVICE.getDeviceModel()
        except Exception:
            return None

    def start_balance(self) -> None:
        if self.robot is None:
            self.connect()
        mode = self.get_chassis_mode()
        if mode != "balance":
            raise HardwareFault(
                "UGOT chassis mode is not 'balance'. Set Self-Balancing Car mode before museum-guide operation."
            )
        try:
            self.robot.balance_start_balancing()
            self._balance_active = True
        except Exception as exc:
            self._balance_active = False
            raise HardwareFault(f"balance_start_balancing failed: {exc}")

    def keep_balance_active(self) -> None:
        if not self._balance_active:
            self.start_balance()

    def command_motion(self, command: MovementCommand) -> None:
        self.keep_balance_active()
        if not self.cfg.allow_physical_movement and command.kind != MotionKind.HOLD:
            raise HardwareCapabilityError("Physical movement is disabled by configuration.")
        try:
            if command.kind == MotionKind.HOLD:
                # Do not call balance_stop_balancing. Attempt a zero-speed balance
                # command so stale forward/reverse commands do not persist.
                try:
                    from ugot import ugot

                    self.robot.balance_move_speed(ugot.E_Model.Direction.forward, 0)
                except Exception:
                    pass
                self._last_motion_t = time.monotonic()
                return
            if command.kind in (MotionKind.FORWARD, MotionKind.BACKWARD):
                from ugot import ugot

                direction = (
                    ugot.E_Model.Direction.forward
                    if command.kind == MotionKind.FORWARD
                    else ugot.E_Model.Direction.backward
                )
                self.robot.balance_move_speed(direction, int(abs(command.speed)))
            elif command.kind in (MotionKind.TURN_LEFT, MotionKind.TURN_RIGHT):
                if self._turn_method is None:
                    raise HardwareCapabilityError("Balance-mode turn API is not verified for this SDK.")
                signed_speed = abs(command.speed) if command.kind == MotionKind.TURN_RIGHT else -abs(command.speed)
                self._turn_method(signed_speed)
            else:
                raise HardwareCapabilityError(f"Unsupported motion kind: {command.kind}")
            self._last_motion_t = time.monotonic()
        except Exception as exc:
            raise HardwareFault(f"Motion command failed while keeping balance active: {exc}")

    def read_distance(self) -> DistanceReading:
        if self._distance_method is None:
            raise HardwareCapabilityError("No verified ultrasonic/distance SDK method is available.")
        try:
            raw = self._distance_method()
            center = _parse_distance_m(raw)
            return DistanceReading(center_m=center, timestamp=time.monotonic(), valid=center is not None, source="ugot")
        except Exception as exc:
            raise HardwareFault(f"Distance read failed: {exc}")

    def read_camera_frame(self) -> PerceptionFrame:
        try:
            frame_bytes = self.robot.read_camera_data()
            valid = frame_bytes is not None
            return PerceptionFrame(timestamp=time.monotonic(), valid=valid, image=frame_bytes)
        except Exception as exc:
            raise HardwareFault(f"Camera read failed: {exc}")

    def read_tilt_deg(self) -> Optional[float]:
        return None

    def speak(self, text: str) -> None:
        if self._tts_method is None:
            raise HardwareCapabilityError("No verified UGOT text-to-speech SDK method is available.")
        try:
            self._tts_method(text)
        except Exception as exc:
            raise HardwareFault(f"TTS failed: {exc}")

    def stop_translation_keep_balance(self) -> None:
        self.command_motion(MovementCommand(MotionKind.HOLD, 0.0, reason="stop translation, keep balance"))

    def emergency_stop(self) -> None:
        # Emergency can disable normal behavior; ordinary decisions must not call this.
        try:
            self.stop_translation_keep_balance()
        finally:
            self._balance_active = True

    def balance_active(self) -> bool:
        return self._balance_active

    def _find_method(self, names: Iterable[str]) -> Optional[Callable[..., Any]]:
        if self.robot is None:
            return None
        for name in names:
            method = getattr(self.robot, name, None)
            if callable(method):
                return method
        return None


class FakeMuseumHardware(MuseumHardware):
    def __init__(self, cfg: Optional[MuseumGuideConfig] = None) -> None:
        self.cfg = cfg or MuseumGuideConfig()
        self._balance_active = False
        self.commands: List[MovementCommand] = []
        self.spoken: List[str] = []
        self.distance_queue: Deque[DistanceReading] = deque()
        self.frame_queue: Deque[PerceptionFrame] = deque()
        self.fail_distance = False
        self.fail_camera = False
        self.fail_speech = False
        self.fail_motion = False
        self.emergency_stopped = False
        self.balance_stop_called = False
        self.tilt_deg: Optional[float] = None

    def start_balance(self) -> None:
        self._balance_active = True

    def keep_balance_active(self) -> None:
        self._balance_active = True

    def command_motion(self, command: MovementCommand) -> None:
        if self.fail_motion:
            raise HardwareFault("fake motion failure")
        self.keep_balance_active()
        self.commands.append(command)

    def read_distance(self) -> DistanceReading:
        if self.fail_distance:
            raise HardwareFault("fake distance failure")
        if self.distance_queue:
            return self.distance_queue.popleft()
        return DistanceReading(center_m=2.0, left_m=2.0, right_m=2.0, timestamp=time.monotonic(), valid=True, source="fake")

    def read_camera_frame(self) -> PerceptionFrame:
        if self.fail_camera:
            raise HardwareFault("fake camera failure")
        if self.frame_queue:
            return self.frame_queue.popleft()
        return PerceptionFrame(timestamp=time.monotonic(), valid=True)

    def read_tilt_deg(self) -> Optional[float]:
        return self.tilt_deg

    def speak(self, text: str) -> None:
        if self.fail_speech:
            raise HardwareFault("fake speech failure")
        self.keep_balance_active()
        self.spoken.append(text)

    def stop_translation_keep_balance(self) -> None:
        self.command_motion(MovementCommand(MotionKind.HOLD, 0.0, reason="fake hold"))

    def emergency_stop(self) -> None:
        self.emergency_stopped = True
        self.command_motion(MovementCommand(MotionKind.HOLD, 0.0, reason="fake emergency hold"))

    def balance_active(self) -> bool:
        return self._balance_active


def _parse_distance_m(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, dict):
        for key in ("m", "meter", "meters", "distance_m", "distance", "cm", "distance_cm"):
            if key in raw:
                value = float(raw[key])
                if key in ("cm", "distance_cm"):
                    value /= 100.0
                break
        else:
            return None
    elif isinstance(raw, (list, tuple)) and raw:
        value = float(raw[0])
    else:
        return None
    if value > 20.0:
        value /= 100.0
    if value <= 0.0:
        return None
    return value

#!/usr/bin/env python3
"""
UGOT Ackermann steering FSM controller for WRO-style obstacle navigation.

This script is written for a 4-wheeled steering robot using a UGOT main
controller, rear drive motors, and a front Ackermann steering actuator.

Assumptions:
    - The laptop / single-board computer runs this Python script.
    - The UGOT controller is reachable over Wi-Fi.
    - A vision/NPU pipeline returns detected_obstacles with:
        color: "red" or "green"
        x_center: normalized 0.0..1.0, or pixel coordinate
        width: normalized 0.0..1.0, or pixel width
        distance_est: optional distance estimate in meters
    - Positive steering angle means steer right.
    - Negative steering angle means steer left.

Install:
    pip install ugot opencv-python numpy

Run:
    python ugot_ackermann_fsm_driver.py --ip 10.196.72.185 --direction CCW
"""

from __future__ import annotations

import argparse
import inspect
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    import grpc
except ImportError:  # pragma: no cover - grpc is installed with the UGOT SDK.
    grpc = None
from ugot import ugot


class State(Enum):
    LANE_FOLLOW = auto()
    AVOID_RED = auto()
    AVOID_GREEN = auto()


class DrivingDirection(str, Enum):
    CW = "CW"
    CCW = "CCW"


class ObstacleColor(str, Enum):
    RED = "red"
    GREEN = "green"


@dataclass(frozen=True)
class DetectedObstacle:
    color: str
    x_center: float
    width: float
    distance_est: Optional[float] = None


@dataclass(frozen=True)
class NormalizedObstacle:
    color: ObstacleColor
    x_center_norm: float
    width_norm: float
    distance_est_m: Optional[float]
    proximity: float


@dataclass
class ControlCommand:
    steering_angle_deg: float
    drive_speed: int
    state: State
    reason: str


@dataclass
class ControlConfig:
    robot_ip: str = "10.196.72.185"
    driving_direction: DrivingDirection = DrivingDirection.CCW

    # Actuator limits. Adjust to your mechanical linkage.
    max_steering_angle_deg: float = 28.0
    neutral_steering_angle_deg: float = 0.0
    cruise_speed: int = 45
    avoid_speed: int = 32
    cautious_speed: int = 24

    # P-controller gains.
    lane_kp: float = 18.0
    obstacle_lateral_kp: float = 34.0
    obstacle_feedforward_deg: float = 10.0

    # Obstacle handling thresholds.
    min_obstacle_width_norm: float = 0.035
    trigger_distance_m: float = 0.95
    close_distance_m: float = 0.35
    avoid_hold_s: float = 0.25

    # Desired image position of a pillar while safely passing it.
    # To pass a red pillar on the robot's right, the pillar should move left
    # in the camera image. To pass a green pillar on the robot's left, the
    # pillar should move right in the camera image.
    red_target_x_norm: float = 0.35
    green_target_x_norm: float = 0.65

    control_period_s: float = 0.05
    show_debug: bool = True

    # Match these to your actual UGOT controller port IDs.
    left_rear_motor_port: int = 1
    right_rear_motor_port: int = 2
    steering_servo_port: int = 3
    steering_servo_neutral_deg: float = 90.0
    steering_servo_inverted: bool = False
    reconnect_delay_s: float = 1.0
    max_reconnect_delay_s: float = 5.0
    display_scale: float = 0.75
    display_every_n_frames: int = 1
    print_stats_every_s: float = 2.0


@dataclass
class RuntimeStats:
    camera_fps: float = 0.0
    display_fps: float = 0.0
    loop_fps: float = 0.0
    frame_age_ms: float = 0.0


class RateMeter:
    def __init__(self, smoothing: float = 0.85) -> None:
        self.smoothing = smoothing
        self.last_t = 0.0
        self.rate = 0.0

    def tick(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        if self.last_t > 0.0:
            dt = max(1e-6, now - self.last_t)
            instant = 1.0 / dt
            self.rate = instant if self.rate <= 0.0 else self.rate * self.smoothing + instant * (1.0 - self.smoothing)
        self.last_t = now
        return self.rate


class UGOTAckermannHardware:
    """
    All UGOT hardware calls are isolated here.

    The official SDK initialization pattern is fixed:
        got = ugot.UGOT()
        got.initialize(ip)
        got.open_camera()

    Motor/servo method names can vary by UGOT firmware, controller mode, and
    port configuration. This class tries the known UGOT motor and servo APIs
    exposed by the installed SDK and reports the available public methods if no
    compatible call exists.
    """

    def __init__(self, cfg: ControlConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.got: Optional[ugot.UGOT] = None
        self._last_command: Optional[ControlCommand] = None
        self._next_reconnect_t = 0.0
        self._camera_fps_meter = RateMeter()
        self.camera_fps = 0.0
        self.last_frame_t = 0.0

    def connect(self) -> None:
        if self.dry_run:
            print("[DRY RUN] Skipping UGOT connection.")
            return

        self.got = ugot.UGOT()
        self.got.initialize(self.cfg.robot_ip)
        self.got.open_camera()
        print(f"[INFO] Connected to UGOT at {self.cfg.robot_ip}")

    def reconnect_if_needed(self) -> None:
        if self.dry_run or self.got is not None or time.monotonic() < self._next_reconnect_t:
            return

        try:
            self.connect()
            if self._last_command is not None:
                self.apply_command(self._last_command)
        except Exception as exc:
            print(f"[WARN] Reconnect failed: {exc}")
            self.got = None
            self._next_reconnect_t = time.monotonic() + self.cfg.reconnect_delay_s

    def read_camera_frame(self) -> Optional[np.ndarray]:
        if self.dry_run or self.got is None:
            return None

        try:
            frame_bytes = self.got.read_camera_data()
        except Exception as exc:
            self._handle_ugot_exception("read_camera_data", exc)
            return None

        if frame_bytes is None:
            self._handle_ugot_exception("read_camera_data", RuntimeError("No camera data received"))
            return None

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            self._handle_ugot_exception("read_camera_data", RuntimeError("Invalid camera frame bytes"))
            return None

        self.last_frame_t = time.monotonic()
        self.camera_fps = self._camera_fps_meter.tick(self.last_frame_t)
        return frame

    def set_drive_speed(self, speed: int) -> None:
        speed = int(clamp(speed, -100, 100))

        if self.dry_run:
            print(f"[DRY RUN] rear drive speed = {speed}")
            return

        if self.got is None:
            return

        try:
            self._set_one_motor_speed(self.cfg.left_rear_motor_port, speed)
            self._set_one_motor_speed(self.cfg.right_rear_motor_port, speed)
        except Exception as exc:
            self._handle_ugot_exception("rear motor speed", exc)

    def set_steering_angle(self, angle_deg: float) -> None:
        angle_deg = clamp(
            angle_deg,
            -self.cfg.max_steering_angle_deg,
            self.cfg.max_steering_angle_deg,
        )

        if self.dry_run:
            print(f"[DRY RUN] steering angle = {angle_deg:.1f} deg")
            return

        if self.got is None:
            return

        servo_delta = -angle_deg if self.cfg.steering_servo_inverted else angle_deg
        servo_angle = clamp(self.cfg.steering_servo_neutral_deg + servo_delta, 0.0, 180.0)

        try:
            self._call_first_available(
                (
                    "set_servo_angle",
                    "servo_set_angle",
                    "servo_control",
                    "set_pwm_servo_angle",
                    "pwm_servo_control",
                    "set_angle_servo",
                ),
                self.cfg.steering_servo_port,
                servo_angle,
            )
        except Exception as exc:
            self._handle_ugot_exception("steering servo angle", exc)

    def apply_command(self, command: ControlCommand) -> None:
        self._last_command = command
        self.reconnect_if_needed()
        self.set_steering_angle(command.steering_angle_deg)
        self.set_drive_speed(command.drive_speed)

    def stop(self) -> None:
        if self.dry_run:
            print("[DRY RUN] stop drive motors")
            return

        if self.got is None:
            return

        try:
            self.got.mecanum_stop()
        except Exception:
            try:
                self._set_one_motor_speed(self.cfg.left_rear_motor_port, 0)
                self._set_one_motor_speed(self.cfg.right_rear_motor_port, 0)
            except Exception as exc:
                print(f"[WARN] Could not stop rear motors: {exc}")

    def _set_one_motor_speed(self, port: int, speed: int) -> None:
        self._call_first_available(
            (
                "set_motor_speed",
                "motor_set_speed",
                "motor_control",
                "set_dc_motor_speed",
                "dc_motor_control",
                "set_motor_power",
            ),
            port,
            speed,
        )

    def _call_first_available(self, names: Sequence[str], *args: Any) -> Any:
        if self.got is None:
            return None

        attempted = []
        for name in names:
            method = getattr(self.got, name, None)
            if method is None:
                continue

            attempted.append(name)
            try:
                return call_sdk_method(method, *args)
            except TypeError as exc:
                attempted.append(f"{name}: {exc}")
                continue

        available = ", ".join(name for name in dir(self.got) if not name.startswith("_"))
        raise RuntimeError(
            "No compatible UGOT SDK method found. "
            f"Tried: {', '.join(attempted or names)}. "
            f"Available public methods: {available}"
        )

    def _handle_ugot_exception(self, operation: str, exc: BaseException) -> None:
        if is_transient_network_error(exc):
            print(f"[WARN] UGOT {operation} interrupted; reconnecting soon: {exc}")
            self.got = None
            self._next_reconnect_t = time.monotonic() + self.cfg.reconnect_delay_s
            return

        print(f"[WARN] UGOT {operation} failed: {exc}")


class NavigationFSM:
    def __init__(self, cfg: ControlConfig) -> None:
        self.cfg = cfg
        self.state = State.LANE_FOLLOW
        self.last_obstacle: Optional[NormalizedObstacle] = None
        self.last_seen_t = 0.0

    def update(
        self,
        detected_obstacles: Sequence[DetectedObstacle],
        frame_shape: Optional[Tuple[int, int, int]] = None,
        lane_center_norm: float = 0.5,
    ) -> ControlCommand:
        now = time.monotonic()
        obstacle = select_priority_obstacle(detected_obstacles, self.cfg, frame_shape)

        if obstacle is not None:
            self.last_obstacle = obstacle
            self.last_seen_t = now
            self.state = State.AVOID_RED if obstacle.color == ObstacleColor.RED else State.AVOID_GREEN
        elif now - self.last_seen_t > self.cfg.avoid_hold_s:
            self.last_obstacle = None
            self.state = State.LANE_FOLLOW

        if self.state == State.LANE_FOLLOW:
            return self._lane_follow(lane_center_norm)

        if self.last_obstacle is None:
            return self._lane_follow(lane_center_norm)

        if self.state == State.AVOID_RED:
            return self._avoid_obstacle(self.last_obstacle, pass_side="right")

        return self._avoid_obstacle(self.last_obstacle, pass_side="left")

    def _lane_follow(self, lane_center_norm: float) -> ControlCommand:
        center_error = clamp(lane_center_norm - 0.5, -1.0, 1.0)
        steering = self.cfg.neutral_steering_angle_deg + self.cfg.lane_kp * center_error
        steering = clamp(steering, -self.cfg.max_steering_angle_deg, self.cfg.max_steering_angle_deg)

        return ControlCommand(
            steering_angle_deg=steering,
            drive_speed=self.cfg.cruise_speed,
            state=State.LANE_FOLLOW,
            reason=f"lane follow, center_error={center_error:+.2f}",
        )

    def _avoid_obstacle(self, obstacle: NormalizedObstacle, pass_side: str) -> ControlCommand:
        if obstacle.color == ObstacleColor.RED:
            target_x = self.cfg.red_target_x_norm
            side_sign = +1.0
        else:
            target_x = self.cfg.green_target_x_norm
            side_sign = -1.0

        # P control: as the pillar gets closer/larger, steering authority rises.
        # Lateral term moves the pillar toward its safe target image location.
        lateral_error = target_x - obstacle.x_center_norm
        proximity_gain = 0.35 + 0.65 * obstacle.proximity
        p_term = -self.cfg.obstacle_lateral_kp * lateral_error * proximity_gain
        feedforward = self.cfg.obstacle_feedforward_deg * side_sign * obstacle.proximity

        steering = self.cfg.neutral_steering_angle_deg + p_term + feedforward
        steering = clamp(steering, -self.cfg.max_steering_angle_deg, self.cfg.max_steering_angle_deg)

        if obstacle.proximity > 0.75:
            speed = self.cfg.cautious_speed
        else:
            speed = self.cfg.avoid_speed

        in_out = inward_outward_label(pass_side, self.cfg.driving_direction)

        return ControlCommand(
            steering_angle_deg=steering,
            drive_speed=speed,
            state=self.state,
            reason=(
                f"{obstacle.color.value} pillar: pass {pass_side} "
                f"({in_out}), x={obstacle.x_center_norm:.2f}, "
                f"target={target_x:.2f}, proximity={obstacle.proximity:.2f}"
            ),
        )


def select_priority_obstacle(
    obstacles: Sequence[DetectedObstacle],
    cfg: ControlConfig,
    frame_shape: Optional[Tuple[int, int, int]],
) -> Optional[NormalizedObstacle]:
    normalized = [
        normalize_obstacle(obstacle, cfg, frame_shape)
        for obstacle in obstacles
        if obstacle.color.lower() in (ObstacleColor.RED.value, ObstacleColor.GREEN.value)
    ]
    actionable = [
        obstacle
        for obstacle in normalized
        if obstacle.width_norm >= cfg.min_obstacle_width_norm or obstacle.proximity > 0.05
    ]

    if not actionable:
        return None

    # Highest proximity wins. This keeps the FSM focused on the immediate risk.
    return max(actionable, key=lambda obstacle: obstacle.proximity)


def normalize_obstacle(
    obstacle: DetectedObstacle,
    cfg: ControlConfig,
    frame_shape: Optional[Tuple[int, int, int]],
) -> NormalizedObstacle:
    frame_width = frame_shape[1] if frame_shape is not None else None

    x_center_norm = normalize_measurement(obstacle.x_center, frame_width)
    width_norm = normalize_measurement(obstacle.width, frame_width)

    if obstacle.distance_est is not None:
        proximity = distance_to_proximity(obstacle.distance_est, cfg)
    else:
        # If distance is unavailable, image width is a useful proxy: bigger
        # bounding box means the pillar is closer.
        proximity = clamp(width_norm / 0.28, 0.0, 1.0)

    return NormalizedObstacle(
        color=ObstacleColor(obstacle.color.lower()),
        x_center_norm=x_center_norm,
        width_norm=width_norm,
        distance_est_m=obstacle.distance_est,
        proximity=proximity,
    )


def normalize_measurement(value: float, image_extent: Optional[int]) -> float:
    if 0.0 <= value <= 1.0:
        return clamp(value, 0.0, 1.0)

    if image_extent is None or image_extent <= 0:
        raise ValueError(
            "Pixel coordinates were provided, but frame_shape is unavailable for normalization."
        )

    return clamp(value / float(image_extent), 0.0, 1.0)


def distance_to_proximity(distance_m: float, cfg: ControlConfig) -> float:
    if distance_m <= cfg.close_distance_m:
        return 1.0
    if distance_m >= cfg.trigger_distance_m:
        return 0.0

    span = cfg.trigger_distance_m - cfg.close_distance_m
    return clamp((cfg.trigger_distance_m - distance_m) / span, 0.0, 1.0)


def inward_outward_label(pass_side: str, direction: DrivingDirection) -> str:
    if pass_side == "right":
        return "outwards" if direction == DrivingDirection.CCW else "inwards"

    return "inwards" if direction == DrivingDirection.CCW else "outwards"


def get_detected_obstacles_from_npu(frame: Optional[np.ndarray]) -> List[DetectedObstacle]:
    """
    Replace this function with your NPU/vision integration.

    Expected return example:
        return [
            DetectedObstacle(color="red", x_center=0.48, width=0.13, distance_est=0.55),
            DetectedObstacle(color="green", x_center=0.72, width=0.08, distance_est=0.85),
        ]

    If your detector returns pixel coordinates, pass frame_shape into
    NavigationFSM.update() and the controller will normalize them.
    """
    _ = frame
    return []


def estimate_lane_center(frame: Optional[np.ndarray]) -> float:
    """
    Placeholder for lane following.

    Return the normalized image x coordinate of the lane center:
        0.5 = centered
        >0.5 = lane center is to the right, steer right
        <0.5 = lane center is to the left, steer left
    """
    _ = frame
    return 0.5


def draw_debug(
    frame: np.ndarray,
    command: ControlCommand,
    obstacles: Sequence[DetectedObstacle],
    stats: Optional[RuntimeStats] = None,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for obstacle in obstacles:
        color = obstacle.color.lower()
        x_norm = normalize_measurement(obstacle.x_center, w)
        width_norm = normalize_measurement(obstacle.width, w)
        x_center = int(x_norm * w)
        box_w = int(width_norm * w)
        x1 = max(0, x_center - box_w // 2)
        x2 = min(w - 1, x_center + box_w // 2)
        y1 = int(h * 0.25)
        y2 = int(h * 0.90)
        bgr = (0, 0, 255) if color == "red" else (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(
            annotated,
            color.upper(),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bgr,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"{command.state.name}: {command.reason}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"steer={command.steering_angle_deg:+.1f} deg speed={command.drive_speed}",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if stats is not None:
        cv2.putText(
            annotated,
            (
                f"cam {stats.camera_fps:4.1f} fps | "
                f"loop {stats.loop_fps:4.1f} fps | "
                f"view {stats.display_fps:4.1f} fps | "
                f"age {stats.frame_age_ms:4.0f} ms"
            ),
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.99:
        return frame
    height, width = frame.shape[:2]
    display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def call_sdk_method(method: Callable[..., Any], *args: Any) -> Any:
    """
    UGOT SDK builds differ by firmware generation. Prefer the natural positional
    call, then retry with the smallest useful prefix when an older method takes
    fewer arguments.
    """
    try:
        return method(*args)
    except TypeError:
        signature = inspect.signature(method)
        required_positionals = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_positionals) < len(args):
            return method(*args[: len(required_positionals)])
        raise


def is_transient_network_error(exc: BaseException) -> bool:
    if grpc is not None and isinstance(exc, grpc.RpcError):
        try:
            return exc.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.CANCELLED,
            )
        except Exception:
            return True

    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "no route to host",
            "host is down",
            "unavailable",
            "connection reset",
            "connection refused",
            "timed out",
            "stream removed",
            "no camera data received",
        )
    )


def run(cfg: ControlConfig, dry_run: bool) -> None:
    hardware = UGOTAckermannHardware(cfg, dry_run=dry_run)
    fsm = NavigationFSM(cfg)
    loop_meter = RateMeter()
    display_meter = RateMeter()
    loop_count = 0
    last_stats_print_t = 0.0

    try:
        hardware.connect()
        print("[INFO] Controller running. Press Ctrl+C to stop.")

        while True:
            loop_count += 1
            now = time.monotonic()
            loop_fps = loop_meter.tick(now)
            hardware.reconnect_if_needed()
            frame = hardware.read_camera_frame()
            frame_shape = frame.shape if frame is not None else None
            frame_age_s = now - hardware.last_frame_t if hardware.last_frame_t > 0.0 else 999.0

            detected_obstacles = get_detected_obstacles_from_npu(frame)
            lane_center_norm = estimate_lane_center(frame)
            command = fsm.update(detected_obstacles, frame_shape, lane_center_norm)

            hardware.apply_command(command)
            print(
                f"[CTRL] {command.state.name:<12} "
                f"steer={command.steering_angle_deg:+05.1f} "
                f"speed={command.drive_speed:03d} "
                f"{command.reason}"
            )

            if cfg.show_debug and frame is not None:
                stats = RuntimeStats(
                    camera_fps=hardware.camera_fps,
                    display_fps=display_meter.rate,
                    loop_fps=loop_fps,
                    frame_age_ms=frame_age_s * 1000.0,
                )
                if loop_count % max(1, cfg.display_every_n_frames) == 0:
                    cv2.imshow(
                        "UGOT Ackermann FSM",
                        resize_for_display(draw_debug(frame, command, detected_obstacles, stats), cfg.display_scale),
                    )
                    display_meter.tick()
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if now - last_stats_print_t >= cfg.print_stats_every_s:
                print(
                    f"[STATS] cam={hardware.camera_fps:.1f}fps "
                    f"loop={loop_fps:.1f}fps view={display_meter.rate:.1f}fps "
                    f"age={frame_age_s * 1000.0:.0f}ms"
                )
                last_stats_print_t = now

            time.sleep(cfg.control_period_s)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping robot.")
    finally:
        hardware.stop()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


def parse_args() -> Tuple[ControlConfig, bool]:
    parser = argparse.ArgumentParser(description="UGOT Ackermann FSM obstacle controller.")
    parser.add_argument("--ip", default="10.196.72.185", help="UGOT controller IP address.")
    parser.add_argument("--direction", choices=("CW", "CCW"), default="CCW", help="Assigned lap direction.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without connecting to UGOT.")
    parser.add_argument("--no-debug", action="store_true", help="Disable OpenCV debug window.")
    parser.add_argument("--display-scale", type=float, default=0.75, help="Scale OpenCV preview window. Lower values reduce display lag.")
    parser.add_argument("--display-every", type=int, default=1, help="Draw every Nth processed frame in the OpenCV window.")
    parser.add_argument("--stats-every", type=float, default=2.0, help="Print FPS/status stats every N seconds.")
    parser.add_argument("--cruise-speed", type=int, default=45)
    parser.add_argument("--avoid-speed", type=int, default=32)
    parser.add_argument("--max-steer", type=float, default=28.0)
    args = parser.parse_args()

    cfg = ControlConfig(
        robot_ip=args.ip,
        driving_direction=DrivingDirection(args.direction),
        cruise_speed=args.cruise_speed,
        avoid_speed=args.avoid_speed,
        max_steering_angle_deg=args.max_steer,
        show_debug=not args.no_debug,
        display_scale=max(0.1, min(1.0, args.display_scale)),
        display_every_n_frames=max(1, args.display_every),
        print_stats_every_s=max(0.5, args.stats_every),
    )
    return cfg, args.dry_run


if __name__ == "__main__":
    run(*parse_args())

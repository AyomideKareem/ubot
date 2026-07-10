#!/usr/bin/env python3
"""
UBTECH uGot WRO Future Engineers brick-aware driving client.

Runs on a laptop connected to the same Wi-Fi network as the uGot robot.
The laptop pulls camera frames from the robot, classifies red/green bricks
with a fast HSV baseline, and sends real-time mecanum drive commands back.

Install:
    pip install ugot opencv-python numpy

Example:
    python ugot_wro_brick_driver.py
    python ugot_wro_brick_driver.py --tune
    python ugot_wro_brick_driver.py --ip 10.196.72.185
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
try:
    import grpc
except ImportError:  # pragma: no cover - grpc is installed with the UGOT SDK.
    grpc = None
from ugot import ugot


class BrickColor(str, Enum):
    RED = "red"
    GREEN = "green"
    NONE = "none"


@dataclass(frozen=True)
class HSVRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


@dataclass
class HSVConfig:
    # Red wraps around hue=0, so use two ranges.
    red_low_1: HSVRange = HSVRange((0, 80, 50), (10, 255, 255))
    red_low_2: HSVRange = HSVRange((170, 80, 50), (179, 255, 255))
    green: HSVRange = HSVRange((35, 60, 45), (90, 255, 255))

    # Ignore tiny blobs/noise. Tune this for camera resolution and distance.
    min_area_px: int = 900

    # Process only the lower portion of the image, where track obstacles appear.
    roi_top_fraction: float = 0.35


@dataclass
class BrickDetection:
    color: BrickColor
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    area_px: int = 0
    center: Optional[Tuple[int, int]] = None


@dataclass
class ControlCommand:
    x_speed: int
    y_speed: int
    z_speed: int
    label: str


@dataclass
class DriverConfig:
    ip: str
    forward_speed: int = 30
    dodge_speed: int = 22
    turn_speed: int = 10
    red_dodge: str = "left"
    control_period_s: float = 0.05
    max_missed_frames: int = 10
    show_window: bool = True
    tune: bool = False
    model_path: Optional[str] = None
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
    missed_frames: int = 0


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


class BrickClassifier:
    """
    Modular classifier wrapper.

    Default backend is HSV thresholding because it is fast, transparent, and
    tuneable at competition venues. To use a learned classifier later, replace
    _classify_with_model() with ONNX Runtime, cv2.dnn, PyTorch, or TensorRT.
    """

    def __init__(self, hsv_config: HSVConfig, model_path: Optional[str] = None) -> None:
        self.hsv_config = hsv_config
        self.model_path = model_path
        self.model = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        # Scaffold for custom inference. For ONNX with OpenCV DNN, for example:
        # self.model = cv2.dnn.readNetFromONNX(model_path)
        # Then implement preprocessing/postprocessing in _classify_with_model().
        self.model = model_path
        print(f"[INFO] Model path supplied: {model_path}")
        print("[INFO] HSV baseline remains active until _classify_with_model() is implemented.")

    def classify_brick(self, frame: np.ndarray) -> BrickDetection:
        if self.model is not None:
            model_detection = self._classify_with_model(frame)
            if model_detection is not None:
                return model_detection

        return self._classify_with_hsv(frame)

    def _classify_with_model(self, frame: np.ndarray) -> Optional[BrickDetection]:
        # Drop-in point for a binary or 3-class model:
        #   input: BGR frame
        #   output: BrickDetection(BrickColor.RED/GREEN/NONE, confidence, bbox, area, center)
        #
        # Returning None intentionally falls back to HSV for production safety.
        _ = frame
        return None

    def _classify_with_hsv(self, frame: np.ndarray) -> BrickDetection:
        cfg = self.hsv_config
        height, width = frame.shape[:2]
        roi_y0 = int(height * cfg.roi_top_fraction)
        roi = frame[roi_y0:height, 0:width]

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        red_mask_1 = mask_from_range(hsv, cfg.red_low_1)
        red_mask_2 = mask_from_range(hsv, cfg.red_low_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        green_mask = mask_from_range(hsv, cfg.green)

        red_detection = best_detection_from_mask(red_mask, BrickColor.RED, cfg.min_area_px, roi_y0)
        green_detection = best_detection_from_mask(green_mask, BrickColor.GREEN, cfg.min_area_px, roi_y0)

        if red_detection.area_px == 0 and green_detection.area_px == 0:
            return BrickDetection(BrickColor.NONE, confidence=0.0)

        if red_detection.area_px >= green_detection.area_px:
            red_detection.confidence = min(1.0, red_detection.area_px / max(1.0, width * height * 0.08))
            return red_detection

        green_detection.confidence = min(1.0, green_detection.area_px / max(1.0, width * height * 0.08))
        return green_detection


def mask_from_range(hsv: np.ndarray, hsv_range: HSVRange) -> np.ndarray:
    lower = np.array(hsv_range.lower, dtype=np.uint8)
    upper = np.array(hsv_range.upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def best_detection_from_mask(
    mask: np.ndarray,
    color: BrickColor,
    min_area_px: int,
    y_offset: int,
) -> BrickDetection:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BrickDetection(BrickColor.NONE, confidence=0.0)

    contour = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(contour))
    if area < min_area_px:
        return BrickDetection(BrickColor.NONE, confidence=0.0)

    x, y, w, h = cv2.boundingRect(contour)
    bbox = (x, y + y_offset, w, h)
    center = (x + w // 2, y + y_offset + h // 2)
    return BrickDetection(color=color, confidence=0.0, bbox=bbox, area_px=area, center=center)


def decide_motion(detection: BrickDetection, cfg: DriverConfig) -> ControlCommand:
    red_x = -cfg.dodge_speed if cfg.red_dodge == "left" else cfg.dodge_speed
    green_x = -red_x

    if detection.color == BrickColor.RED:
        return ControlCommand(
            x_speed=red_x,
            y_speed=max(10, cfg.forward_speed - 8),
            z_speed=-cfg.turn_speed if red_x < 0 else cfg.turn_speed,
            label="RED BRICK: DODGE",
        )

    if detection.color == BrickColor.GREEN:
        return ControlCommand(
            x_speed=green_x,
            y_speed=max(10, cfg.forward_speed - 8),
            z_speed=-cfg.turn_speed if green_x < 0 else cfg.turn_speed,
            label="GREEN BRICK: DODGE",
        )

    return ControlCommand(
        x_speed=0,
        y_speed=cfg.forward_speed,
        z_speed=0,
        label="NO BRICK: FORWARD",
    )


def decode_camera_frame(frame_bytes: Optional[bytes]) -> Optional[np.ndarray]:
    if frame_bytes is None:
        return None

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None

    return frame


def draw_debug_overlay(
    frame: np.ndarray,
    detection: BrickDetection,
    command: ControlCommand,
    stats: Optional[RuntimeStats] = None,
) -> np.ndarray:
    annotated = frame.copy()
    color_bgr = {
        BrickColor.RED: (0, 0, 255),
        BrickColor.GREEN: (0, 200, 0),
        BrickColor.NONE: (220, 220, 220),
    }[detection.color]

    if detection.bbox is not None:
        x, y, w, h = detection.bbox
        label = "RED BRICK" if detection.color == BrickColor.RED else "GREEN BRICK"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.putText(
            annotated,
            f"{label} {detection.confidence:.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        if detection.center is not None:
            cv2.circle(annotated, detection.center, 5, color_bgr, -1)

    cv2.putText(
        annotated,
        command.label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"x={command.x_speed} y={command.y_speed} z={command.z_speed}",
        (16, 64),
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
                f"age {stats.frame_age_ms:4.0f} ms | "
                f"miss {stats.missed_frames}"
            ),
            (16, 94),
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


def create_tuning_window() -> None:
    cv2.namedWindow("HSV Tuning", cv2.WINDOW_NORMAL)
    sliders = {
        "R1 H min": 0,
        "R1 H max": 10,
        "R2 H min": 170,
        "R2 H max": 179,
        "G H min": 35,
        "G H max": 90,
        "S min": 60,
        "V min": 45,
        "Min area": 900,
    }

    for name, value in sliders.items():
        max_value = 5000 if name == "Min area" else 255
        if "H " in name:
            max_value = 179
        cv2.createTrackbar(name, "HSV Tuning", value, max_value, lambda _: None)


def update_hsv_config_from_sliders(cfg: HSVConfig) -> HSVConfig:
    s_min = cv2.getTrackbarPos("S min", "HSV Tuning")
    v_min = cv2.getTrackbarPos("V min", "HSV Tuning")
    min_area = max(1, cv2.getTrackbarPos("Min area", "HSV Tuning"))

    cfg.red_low_1 = HSVRange(
        (cv2.getTrackbarPos("R1 H min", "HSV Tuning"), s_min, v_min),
        (cv2.getTrackbarPos("R1 H max", "HSV Tuning"), 255, 255),
    )
    cfg.red_low_2 = HSVRange(
        (cv2.getTrackbarPos("R2 H min", "HSV Tuning"), s_min, v_min),
        (cv2.getTrackbarPos("R2 H max", "HSV Tuning"), 255, 255),
    )
    cfg.green = HSVRange(
        (cv2.getTrackbarPos("G H min", "HSV Tuning"), s_min, v_min),
        (cv2.getTrackbarPos("G H max", "HSV Tuning"), 255, 255),
    )
    cfg.min_area_px = min_area
    return cfg


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
        )
    )


def connect_robot(ip: str, open_camera: bool = True) -> ugot.UGOT:
    got = ugot.UGOT()
    got.initialize(ip)
    if open_camera:
        got.open_camera()
    return got


class CameraWorker:
    def __init__(self, cfg: DriverConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_t = 0.0
        self._last_error: Optional[BaseException] = None
        self._got: Optional[ugot.UGOT] = None
        self._fps_meter = RateMeter()
        self._camera_fps = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ugot-camera-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> Tuple[Optional[np.ndarray], float, Optional[BaseException], float]:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return frame, self._last_frame_t, self._last_error, self._camera_fps

    def _run(self) -> None:
        delay_s = self.cfg.reconnect_delay_s

        while not self._stop_event.is_set():
            try:
                if self._got is None:
                    print(f"[INFO] Opening camera stream at {self.cfg.ip}")
                    self._got = connect_robot(self.cfg.ip, open_camera=True)
                    delay_s = self.cfg.reconnect_delay_s

                frame = decode_camera_frame(self._got.read_camera_data())
                if frame is None:
                    raise RuntimeError("No camera data received")

                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_t = time.monotonic()
                    self._camera_fps = self._fps_meter.tick(self._last_frame_t)
                    self._last_error = None

            except Exception as exc:
                with self._lock:
                    self._last_error = exc

                if not is_transient_network_error(exc) and str(exc) != "No camera data received":
                    print(f"[WARN] Camera read failed: {exc}")
                else:
                    print(f"[WARN] Camera stream interrupted: {exc}")

                self._got = None
                self._stop_event.wait(delay_s)
                delay_s = min(self.cfg.max_reconnect_delay_s, delay_s * 1.5)


class MotionClient:
    def __init__(self, cfg: DriverConfig) -> None:
        self.cfg = cfg
        self.got: Optional[ugot.UGOT] = None
        self._last_command: Optional[ControlCommand] = None
        self._next_reconnect_t = 0.0

    def connect(self) -> None:
        print(f"[INFO] Connecting motion client to uGot robot at {self.cfg.ip}")
        self.got = connect_robot(self.cfg.ip, open_camera=False)

    def reconnect_if_due(self) -> None:
        if self.got is not None or time.monotonic() < self._next_reconnect_t:
            return

        try:
            self.connect()
            if self._last_command is not None:
                self.move(self._last_command)
        except Exception as exc:
            print(f"[WARN] Motion reconnect failed: {exc}")
            self.got = None
            self._next_reconnect_t = time.monotonic() + self.cfg.reconnect_delay_s

    def move(self, command: ControlCommand) -> None:
        self._last_command = command
        if self.got is None:
            self.reconnect_if_due()
            return

        try:
            # UGOT SDK releases commonly expose mecanum_move_xyz as a positional
            # method. Keyword arguments crash on those builds.
            self.got.mecanum_move_xyz(command.x_speed, command.y_speed, command.z_speed)
        except Exception as exc:
            if is_transient_network_error(exc):
                print(f"[WARN] Motion command lost; will reconnect: {exc}")
                self.got = None
                self._next_reconnect_t = time.monotonic() + self.cfg.reconnect_delay_s
                return
            raise

    def stop(self) -> None:
        if self.got is None:
            return

        try:
            self.got.mecanum_stop()
            print("[INFO] Chassis stopped.")
        except Exception as exc:
            print(f"[WARN] Could not send mecanum_stop(): {exc}")


def run_driver(cfg: DriverConfig) -> None:
    hsv_config = HSVConfig()
    classifier = BrickClassifier(hsv_config=hsv_config, model_path=cfg.model_path)
    camera = CameraWorker(cfg)
    motion = MotionClient(cfg)
    missed_frames = 0
    loop_meter = RateMeter()
    display_meter = RateMeter()
    loop_count = 0
    last_stats_print_t = 0.0

    if cfg.tune:
        create_tuning_window()

    try:
        camera.start()
        motion.connect()
        print("[INFO] Camera and motion clients are running. Press 'q' in the video window or Ctrl+C to stop.")

        while True:
            loop_count += 1
            now = time.monotonic()
            loop_fps = loop_meter.tick(now)
            frame, last_frame_t, camera_error, camera_fps = camera.snapshot()
            frame_age_s = now - last_frame_t if last_frame_t > 0.0 else 999.0
            stale_frame = last_frame_t == 0.0 or frame_age_s > 0.75

            if frame is None or stale_frame:
                missed_frames += 1
                if camera_error is not None:
                    print(f"[WARN] Camera unavailable: {camera_error}")
                print(f"[WARN] Missed/invalid camera frame ({missed_frames}/{cfg.max_missed_frames})")
                if missed_frames >= cfg.max_missed_frames:
                    print("[WARN] Too many missed frames; stopping chassis until stream recovers.")
                    motion.stop()
                    missed_frames = 0
                time.sleep(cfg.control_period_s)
                continue

            missed_frames = 0

            if cfg.tune:
                hsv_config = update_hsv_config_from_sliders(hsv_config)
                classifier.hsv_config = hsv_config

            detection = classifier.classify_brick(frame)
            command = decide_motion(detection, cfg)

            motion.move(command)

            if cfg.show_window:
                stats = RuntimeStats(
                    camera_fps=camera_fps,
                    display_fps=display_meter.rate,
                    loop_fps=loop_fps,
                    frame_age_ms=frame_age_s * 1000.0,
                    missed_frames=missed_frames,
                )
                if loop_count % max(1, cfg.display_every_n_frames) == 0:
                    annotated = draw_debug_overlay(frame, detection, command, stats)
                    cv2.imshow("uGot WRO Brick Driver", resize_for_display(annotated, cfg.display_scale))
                    display_meter.tick()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit requested from OpenCV window.")
                    break

            if now - last_stats_print_t >= cfg.print_stats_every_s:
                print(
                    f"[STATS] cam={camera_fps:.1f}fps "
                    f"loop={loop_fps:.1f}fps view={display_meter.rate:.1f}fps "
                    f"age={frame_age_s * 1000.0:.0f}ms missed={missed_frames}"
                )
                last_stats_print_t = now

            time.sleep(cfg.control_period_s)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping robot.")
    finally:
        camera.stop()
        motion.stop()

        cv2.destroyAllWindows()
        print("[INFO] OpenCV windows closed.")


def parse_args() -> DriverConfig:
    parser = argparse.ArgumentParser(
        description="Laptop-side WRO Future Engineers brick detector and uGot driver."
    )
    parser.add_argument(
        "--ip",
        default="192.168.1.77",
        help="uGot robot local IP address. Defaults to 10.196.72.185.",
    )
    parser.add_argument("--forward-speed", type=int, default=30, help="Forward y_speed while driving clear track.")
    parser.add_argument("--dodge-speed", type=int, default=22, help="Strafe x_speed when dodging a detected brick.")
    parser.add_argument("--turn-speed", type=int, default=10, help="z_speed rotation paired with dodge motion.")
    parser.add_argument(
        "--red-dodge",
        choices=("left", "right"),
        default="left",
        help="Direction to strafe when a red brick is detected. Green uses the opposite direction.",
    )
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.05,
        help="Sleep time per loop in seconds. Increase if Wi-Fi or CPU becomes overloaded.",
    )
    parser.add_argument("--max-missed-frames", type=int, default=10, help="Stop after this many bad frames.")
    parser.add_argument("--no-window", action="store_true", help="Disable local OpenCV debug display.")
    parser.add_argument("--display-scale", type=float, default=0.75, help="Scale OpenCV preview window. Lower values reduce display lag.")
    parser.add_argument("--display-every", type=int, default=1, help="Draw every Nth processed frame in the OpenCV window.")
    parser.add_argument("--stats-every", type=float, default=2.0, help="Print FPS/status stats every N seconds.")
    parser.add_argument("--tune", action="store_true", help="Show HSV trackbars for venue lighting calibration.")
    parser.add_argument("--model-path", default=None, help="Optional ONNX/PyTorch model path scaffold.")
    args = parser.parse_args()

    return DriverConfig(
        ip=args.ip,
        forward_speed=args.forward_speed,
        dodge_speed=args.dodge_speed,
        turn_speed=args.turn_speed,
        red_dodge=args.red_dodge,
        control_period_s=args.control_period,
        max_missed_frames=args.max_missed_frames,
        show_window=not args.no_window,
        tune=args.tune,
        model_path=args.model_path,
        display_scale=max(0.1, min(1.0, args.display_scale)),
        display_every_n_frames=max(1, args.display_every),
        print_stats_every_s=max(0.5, args.stats_every),
    )


if __name__ == "__main__":
    run_driver(parse_args())

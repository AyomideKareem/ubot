#!/usr/bin/env python3
"""Guarded custom balance-control lab for UGOT two-wheel builds.

Use this only to diagnose sensors, motor signs, and timing. The normal
self-balancing path in this repo uses UGOT's built-in balance-car firmware API.
Do not run this custom PID while the robot is configured in UGOT balance mode.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ugot import ugot


@dataclass
class BalanceConfig:
    ip: str = "192.168.1.77"

    # Keep all balance-loop tuning and safety values in one place.
    loop_hz: float = 100.0
    alpha: float = 0.98
    target_pitch_deg: float = 0.0
    kp: float = 2.0
    ki: float = 0.0
    kd: float = 0.25
    integral_limit: float = 20.0
    output_limit: float = 18.0
    output_slew_per_s: float = 80.0
    safe_tilt_deg: float = 18.0
    max_dt_s: float = 0.05
    telemetry_hz: float = 10.0
    slow_loop_warn_ms: float = 20.0

    # Sensor sign/axis mapping. Adjust only after running --axis-test.
    accel_pitch_axis: str = "x"
    accel_pitch_sign: float = 1.0
    gyro_pitch_axis: str = "y"
    gyro_pitch_sign: float = 1.0
    gyro_units: str = "deg_s"  # deg_s or rad_s

    # Calibration constraints.
    calibration_samples: int = 250
    calibration_hz: float = 100.0
    max_accel_std_g: float = 0.08
    max_gyro_std_deg_s: float = 3.0

    # Two-wheel motor mapping. Mirrored installations often need opposite signs.
    left_motor_port: int = 1
    right_motor_port: int = 3
    left_motor_sign: int = 1
    right_motor_sign: int = 1

    # Direction test limit; intentionally low power.
    direction_test_power: float = 8.0
    direction_test_pulse_s: float = 0.15


@dataclass
class ImuSample:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class Calibration:
    gyro_bias: Dict[str, float]
    upright_pitch_offset_deg: float
    accel_std_g: Dict[str, float]
    gyro_std_deg_s: Dict[str, float]


@dataclass
class PidTerms:
    error: float
    p: float
    i: float
    d: float
    unclamped_output: float
    motor_output: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def axis_value(sample: ImuSample, axis: str, prefix: str) -> float:
    return getattr(sample, prefix + axis)


class SdkAdapter:
    def __init__(self, robot: ugot.UGOT, cfg: BalanceConfig) -> None:
        self.robot = robot
        self.cfg = cfg
        self._imu_method = self._find_method(
            (
                "read_imu",
                "get_imu",
                "get_imu_data",
                "read_imu_data",
                "get_motion_data",
                "read_motion_data",
                "get_acceleration_gyro",
                "get_accelerometer_gyroscope",
            )
        )
        self._accel_method = self._find_method(
            (
                "read_accelerometer",
                "get_accelerometer",
                "get_acceleration",
                "read_acceleration",
                "get_accelerate",
            )
        )
        self._gyro_method = self._find_method(
            (
                "read_gyroscope",
                "get_gyroscope",
                "get_gyro",
                "read_gyro",
                "get_angular_velocity",
            )
        )
        self._motor_method = self._find_method(
            (
                "set_motor_speed",
                "motor_set_speed",
                "motor_control",
                "set_dc_motor_speed",
                "dc_motor_control",
                "set_motor_power",
                "turn_motor_speed",
            )
        )

    def public_methods(self) -> List[str]:
        return [name for name in dir(self.robot) if not name.startswith("_")]

    def supports_custom_balance(self) -> bool:
        return self.can_read_imu() and self._motor_method is not None

    def can_read_imu(self) -> bool:
        return self._imu_method is not None or (
            self._accel_method is not None and self._gyro_method is not None
        )

    def can_drive_motors(self) -> bool:
        return self._motor_method is not None

    def read_imu(self) -> ImuSample:
        if self._imu_method is not None:
            return parse_imu_payload(call_sdk_method(self._imu_method))
        if self._accel_method is not None and self._gyro_method is not None:
            accel = parse_vector_payload(call_sdk_method(self._accel_method), ("ax", "ay", "az"))
            gyro = parse_vector_payload(call_sdk_method(self._gyro_method), ("gx", "gy", "gz"))
            return ImuSample(
                ax=accel["ax"],
                ay=accel["ay"],
                az=accel["az"],
                gx=gyro["gx"],
                gy=gyro["gy"],
                gz=gyro["gz"],
            )
        raise RuntimeError("No compatible IMU/accelerometer/gyro read methods found in UGOT SDK.")

    def set_motor_output(self, output: float) -> None:
        if self._motor_method is None:
            raise RuntimeError("No compatible motor-speed method found in UGOT SDK.")
        left = int(clamp(output * self.cfg.left_motor_sign, -self.cfg.output_limit, self.cfg.output_limit))
        right = int(clamp(output * self.cfg.right_motor_sign, -self.cfg.output_limit, self.cfg.output_limit))
        call_sdk_method(self._motor_method, self.cfg.left_motor_port, left)
        call_sdk_method(self._motor_method, self.cfg.right_motor_port, right)

    def stop_motors(self) -> None:
        if self._motor_method is None:
            return
        try:
            call_sdk_method(self._motor_method, self.cfg.left_motor_port, 0)
            call_sdk_method(self._motor_method, self.cfg.right_motor_port, 0)
        except Exception as exc:
            print(f"[WARN] Could not stop motors with custom adapter: {exc}")

    def _find_method(self, names: Iterable[str]) -> Optional[Callable[..., Any]]:
        for name in names:
            method = getattr(self.robot, name, None)
            if callable(method):
                return method
        return None


def call_sdk_method(method: Callable[..., Any], *args: Any) -> Any:
    return method(*args)


def parse_imu_payload(payload: Any) -> ImuSample:
    if isinstance(payload, ImuSample):
        return payload
    if isinstance(payload, dict):
        flat = flatten_dict(payload)
        return ImuSample(
            ax=value_from_keys(flat, ("ax", "accel_x", "accelerometer_x", "x_accel", "acc_x")),
            ay=value_from_keys(flat, ("ay", "accel_y", "accelerometer_y", "y_accel", "acc_y")),
            az=value_from_keys(flat, ("az", "accel_z", "accelerometer_z", "z_accel", "acc_z")),
            gx=value_from_keys(flat, ("gx", "gyro_x", "gyroscope_x", "x_gyro")),
            gy=value_from_keys(flat, ("gy", "gyro_y", "gyroscope_y", "y_gyro")),
            gz=value_from_keys(flat, ("gz", "gyro_z", "gyroscope_z", "z_gyro")),
        )
    if isinstance(payload, (list, tuple)) and len(payload) >= 6:
        return ImuSample(
            ax=float(payload[0]),
            ay=float(payload[1]),
            az=float(payload[2]),
            gx=float(payload[3]),
            gy=float(payload[4]),
            gz=float(payload[5]),
        )
    raise RuntimeError(f"Unsupported IMU payload format: {payload!r}")


def parse_vector_payload(payload: Any, keys: Tuple[str, str, str]) -> Dict[str, float]:
    if isinstance(payload, dict):
        flat = flatten_dict(payload)
        return {
            keys[0]: value_from_keys(flat, (keys[0], "x")),
            keys[1]: value_from_keys(flat, (keys[1], "y")),
            keys[2]: value_from_keys(flat, (keys[2], "z")),
        }
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        return {keys[0]: float(payload[0]), keys[1]: float(payload[1]), keys[2]: float(payload[2])}
    raise RuntimeError(f"Unsupported vector payload format: {payload!r}")


def flatten_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        key_s = str(key).lower()
        if isinstance(value, dict):
            for child_key, child_value in flatten_dict(value).items():
                flat[f"{key_s}_{child_key}"] = child_value
                flat[child_key] = child_value
        else:
            flat[key_s] = value
    return flat


def value_from_keys(payload: Dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in payload:
            return float(payload[key])
    raise RuntimeError(f"Missing expected IMU key. Tried {keys}; payload keys are {sorted(payload)}")


def accelerometer_pitch(sample: ImuSample, cfg: BalanceConfig) -> float:
    pitch_axis = axis_value(sample, cfg.accel_pitch_axis, "a") * cfg.accel_pitch_sign
    other_axes = [axis for axis in ("x", "y", "z") if axis != cfg.accel_pitch_axis]
    denom = math.sqrt(sum(axis_value(sample, axis, "a") ** 2 for axis in other_axes))
    return math.degrees(math.atan2(pitch_axis, max(1e-9, denom)))


def gyro_pitch_rate(sample: ImuSample, cfg: BalanceConfig, calibration: Calibration) -> float:
    raw = axis_value(sample, cfg.gyro_pitch_axis, "g")
    corrected = (raw - calibration.gyro_bias[cfg.gyro_pitch_axis]) * cfg.gyro_pitch_sign
    if cfg.gyro_units == "rad_s":
        corrected = math.degrees(corrected)
    return corrected


def calibrate_stationary(adapter: SdkAdapter, cfg: BalanceConfig) -> Calibration:
    print("[CAL] Keep the robot upright and completely still.")
    input("[CAL] Press Enter when it is stationary.")

    samples: List[ImuSample] = []
    period = 1.0 / cfg.calibration_hz
    next_t = time.monotonic()
    for idx in range(cfg.calibration_samples):
        samples.append(adapter.read_imu())
        next_t += period
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        if (idx + 1) % 50 == 0:
            print(f"[CAL] collected {idx + 1}/{cfg.calibration_samples} samples")

    axes = ("x", "y", "z")
    accel_values = {axis: [axis_value(s, axis, "a") for s in samples] for axis in axes}
    gyro_values = {axis: [axis_value(s, axis, "g") for s in samples] for axis in axes}
    accel_std = {axis: stddev(values) for axis, values in accel_values.items()}
    gyro_std = {axis: stddev(values) for axis, values in gyro_values.items()}
    if cfg.gyro_units == "rad_s":
        gyro_std_deg = {axis: math.degrees(value) for axis, value in gyro_std.items()}
    else:
        gyro_std_deg = gyro_std

    if max(accel_std.values()) > cfg.max_accel_std_g:
        raise RuntimeError(f"Calibration rejected: accelerometer variation too high: {accel_std}")
    if max(gyro_std_deg.values()) > cfg.max_gyro_std_deg_s:
        raise RuntimeError(f"Calibration rejected: gyro variation too high: {gyro_std_deg}")

    gyro_bias = {axis: mean(values) for axis, values in gyro_values.items()}
    upright_pitch = mean([accelerometer_pitch(sample, cfg) for sample in samples])
    calibration = Calibration(
        gyro_bias=gyro_bias,
        upright_pitch_offset_deg=upright_pitch,
        accel_std_g=accel_std,
        gyro_std_deg_s=gyro_std_deg,
    )
    print(f"[CAL] gyro_bias={calibration.gyro_bias}")
    print(f"[CAL] upright_pitch_offset_deg={calibration.upright_pitch_offset_deg:.3f}")
    print(f"[CAL] accel_std_g={calibration.accel_std_g}")
    print(f"[CAL] gyro_std_deg_s={calibration.gyro_std_deg_s}")
    return calibration


class BalancePid:
    def __init__(self, cfg: BalanceConfig) -> None:
        self.cfg = cfg
        self.integral = 0.0
        self.last_output = 0.0

    def update(self, measured_pitch: float, gyro_rate: float, dt: float) -> PidTerms:
        error = self.cfg.target_pitch_deg - measured_pitch
        self.integral = clamp(
            self.integral + error * dt,
            -self.cfg.integral_limit,
            self.cfg.integral_limit,
        )
        p = self.cfg.kp * error
        i = self.cfg.ki * self.integral
        d = self.cfg.kd * (-gyro_rate)
        unclamped = p + i + d
        limited = clamp(unclamped, -self.cfg.output_limit, self.cfg.output_limit)
        max_delta = self.cfg.output_slew_per_s * dt
        output = clamp(limited, self.last_output - max_delta, self.last_output + max_delta)
        self.last_output = output
        return PidTerms(error=error, p=p, i=i, d=d, unclamped_output=unclamped, motor_output=output)


class Telemetry:
    def __init__(self, cfg: BalanceConfig, csv_path: Optional[str]) -> None:
        self.cfg = cfg
        self.period = 1.0 / cfg.telemetry_hz
        self.next_t = 0.0
        self.file_obj = open(csv_path, "w", newline="") if csv_path else None
        self.writer = csv.writer(self.file_obj) if self.file_obj else None
        self.printed_header = False
        self.header = (
            "timestamp",
            "dt",
            "raw_ax",
            "raw_ay",
            "raw_az",
            "raw_gx",
            "raw_gy",
            "raw_gz",
            "filtered_pitch",
            "target_pitch",
            "error",
            "P",
            "I",
            "D",
            "unclamped_output",
            "motor_output",
            "safety_state",
        )
        if self.writer:
            self.writer.writerow(self.header)

    def emit(
        self,
        now: float,
        dt: float,
        sample: ImuSample,
        filtered_pitch: float,
        terms: PidTerms,
        safety_state: str,
    ) -> None:
        if now < self.next_t:
            return
        self.next_t = now + self.period
        row = (
            f"{now:.6f}",
            f"{dt:.6f}",
            f"{sample.ax:.6f}",
            f"{sample.ay:.6f}",
            f"{sample.az:.6f}",
            f"{sample.gx:.6f}",
            f"{sample.gy:.6f}",
            f"{sample.gz:.6f}",
            f"{filtered_pitch:.6f}",
            f"{self.cfg.target_pitch_deg:.6f}",
            f"{terms.error:.6f}",
            f"{terms.p:.6f}",
            f"{terms.i:.6f}",
            f"{terms.d:.6f}",
            f"{terms.unclamped_output:.6f}",
            f"{terms.motor_output:.6f}",
            safety_state,
        )
        if self.writer:
            self.writer.writerow(row)
        else:
            if not self.printed_header:
                print(",".join(self.header))
                self.printed_header = True
            print(",".join(row))

    def close(self) -> None:
        if self.file_obj:
            self.file_obj.close()


def get_chassis_mode(robot: ugot.UGOT) -> Optional[str]:
    try:
        return robot.DEVICE.getDeviceModel()
    except Exception as exc:
        print(f"[WARN] Could not read chassis mode: {exc}")
        return None


def connect(cfg: BalanceConfig) -> ugot.UGOT:
    robot = ugot.UGOT()
    robot.initialize(cfg.ip)
    return robot


def require_not_builtin_balance(robot: ugot.UGOT, force: bool) -> None:
    mode = get_chassis_mode(robot)
    if mode is not None:
        print(f"[INFO] Chassis mode: {mode}")
    if mode == "balance" and not force:
        raise RuntimeError(
            "Refusing custom motor PID while UGOT reports built-in balance mode. "
            "Use twowheeled.py/ugot_balance_diagnostics.py for firmware balance, "
            "or pass --force-custom-in-balance-mode only if you understand the risk."
        )


def run_capability_probe(adapter: SdkAdapter) -> None:
    print("[SDK] Public methods:")
    for name in adapter.public_methods():
        print(f"  {name}")
    print(f"[SDK] IMU readable: {adapter.can_read_imu()}")
    print(f"[SDK] Motor command available: {adapter.can_drive_motors()}")
    print(f"[SDK] Custom balance possible: {adapter.supports_custom_balance()}")


def run_axis_test(adapter: SdkAdapter, cfg: BalanceConfig, calibration: Calibration) -> None:
    print("[AXIS] This test does not drive motors.")
    print("[AXIS] Tilt the robot forward, then backward. Watch pitch/rate signs.")
    start = time.monotonic()
    while time.monotonic() - start < 12.0:
        sample = adapter.read_imu()
        pitch = accelerometer_pitch(sample, cfg) - calibration.upright_pitch_offset_deg
        rate = gyro_pitch_rate(sample, cfg, calibration)
        print(f"[AXIS] pitch={pitch:+7.3f} deg gyro_rate={rate:+7.3f} deg/s", end="\r")
        time.sleep(0.1)
    print()
    print("[AXIS] Configure accel_pitch_sign/gyro_pitch_sign so forward tilt has the expected sign for your controller.")


def run_direction_test(adapter: SdkAdapter, cfg: BalanceConfig) -> None:
    if not adapter.can_drive_motors():
        raise RuntimeError("Cannot run direction test: no compatible motor command method found.")
    print("[DIR] Low-power pulse test. Lift wheels off the floor or support the robot.")
    print(f"[DIR] Pulsing +{cfg.direction_test_power} then -{cfg.direction_test_power} for {cfg.direction_test_pulse_s}s.")
    input("[DIR] Press Enter to pulse motors.")
    try:
        adapter.set_motor_output(cfg.direction_test_power)
        time.sleep(cfg.direction_test_pulse_s)
        adapter.stop_motors()
        time.sleep(0.5)
        adapter.set_motor_output(-cfg.direction_test_power)
        time.sleep(cfg.direction_test_pulse_s)
    finally:
        adapter.stop_motors()
    print("[DIR] For a forward tilt, the corrective wheel torque must move the wheels forward under the center of mass.")
    print("[DIR] If the pulse direction is wrong, flip left_motor_sign/right_motor_sign together or fix mirrored motor wiring.")


def run_balance_loop(adapter: SdkAdapter, cfg: BalanceConfig, calibration: Calibration, csv_path: Optional[str]) -> None:
    if not adapter.supports_custom_balance():
        raise RuntimeError("Cannot run custom balance: SDK lacks compatible IMU or motor methods.")

    pid = BalancePid(cfg)
    telemetry = Telemetry(cfg, csv_path)
    period = 1.0 / cfg.loop_hz
    missed = 0
    safety_state = "starting"
    last_t = time.monotonic()
    sample = adapter.read_imu()
    filtered_pitch = accelerometer_pitch(sample, cfg) - calibration.upright_pitch_offset_deg
    next_t = last_t + period

    print("[RUN] Custom PID loop started. Keep one hand ready to catch the robot.")
    try:
        while True:
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt <= 0.0 or dt > cfg.max_dt_s:
                dt = period
                missed += 1

            sample = adapter.read_imu()
            accel_pitch = accelerometer_pitch(sample, cfg) - calibration.upright_pitch_offset_deg
            rate = gyro_pitch_rate(sample, cfg, calibration)
            filtered_pitch = cfg.alpha * (filtered_pitch + rate * dt) + (1.0 - cfg.alpha) * accel_pitch
            terms = pid.update(filtered_pitch, rate, dt)

            if not math.isfinite(filtered_pitch) or not math.isfinite(rate):
                safety_state = "invalid_imu"
                adapter.stop_motors()
                raise RuntimeError("Invalid IMU value")
            if abs(filtered_pitch) > cfg.safe_tilt_deg:
                safety_state = "tilt_limit"
                adapter.stop_motors()
                raise RuntimeError(f"Tilt limit exceeded: {filtered_pitch:.2f} deg")

            safety_state = "ok"
            adapter.set_motor_output(terms.motor_output)
            telemetry.emit(now, dt, sample, filtered_pitch, terms, safety_state)

            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                missed += 1
                if -sleep_s * 1000.0 > cfg.slow_loop_warn_ms:
                    print(f"[WARN] Slow balance loop: {-sleep_s * 1000.0:.1f}ms late, missed={missed}")
            next_t += period
    except KeyboardInterrupt:
        print("\n[RUN] Interrupted.")
    except Exception as exc:
        print(f"\n[SAFETY] {safety_state}: {exc}")
    finally:
        adapter.stop_motors()
        telemetry.close()
        print(f"[RUN] Motors stopped. Missed/slow iterations: {missed}")


def parse_args() -> Tuple[argparse.Namespace, BalanceConfig]:
    parser = argparse.ArgumentParser(description="Guarded UGOT custom balance diagnostics")
    parser.add_argument("--ip", default=BalanceConfig.ip)
    parser.add_argument("--probe-sdk", action="store_true", help="List SDK methods and detected IMU/motor support.")
    parser.add_argument("--calibrate", action="store_true", help="Run stationary IMU calibration only.")
    parser.add_argument("--axis-test", action="store_true", help="Run pitch-axis/sign diagnostic without motor output.")
    parser.add_argument("--direction-test", action="store_true", help="Run low-power motor direction pulse test.")
    parser.add_argument("--run-custom-pid", action="store_true", help="Run the guarded custom PID loop.")
    parser.add_argument("--force-custom-in-balance-mode", action="store_true")
    parser.add_argument("--csv", default=None, help="Optional CSV telemetry path.")
    parser.add_argument("--loop-hz", type=float, default=BalanceConfig.loop_hz)
    parser.add_argument("--kp", type=float, default=BalanceConfig.kp)
    parser.add_argument("--ki", type=float, default=BalanceConfig.ki)
    parser.add_argument("--kd", type=float, default=BalanceConfig.kd)
    parser.add_argument("--target-pitch", type=float, default=BalanceConfig.target_pitch_deg)
    parser.add_argument("--output-limit", type=float, default=BalanceConfig.output_limit)
    parser.add_argument("--left-sign", type=int, choices=(-1, 1), default=BalanceConfig.left_motor_sign)
    parser.add_argument("--right-sign", type=int, choices=(-1, 1), default=BalanceConfig.right_motor_sign)
    parser.add_argument("--accel-axis", choices=("x", "y", "z"), default=BalanceConfig.accel_pitch_axis)
    parser.add_argument("--accel-sign", type=float, choices=(-1.0, 1.0), default=BalanceConfig.accel_pitch_sign)
    parser.add_argument("--gyro-axis", choices=("x", "y", "z"), default=BalanceConfig.gyro_pitch_axis)
    parser.add_argument("--gyro-sign", type=float, choices=(-1.0, 1.0), default=BalanceConfig.gyro_pitch_sign)
    parser.add_argument("--gyro-units", choices=("deg_s", "rad_s"), default=BalanceConfig.gyro_units)
    args = parser.parse_args()

    cfg = BalanceConfig(
        ip=args.ip,
        loop_hz=args.loop_hz,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        target_pitch_deg=args.target_pitch,
        output_limit=args.output_limit,
        left_motor_sign=args.left_sign,
        right_motor_sign=args.right_sign,
        accel_pitch_axis=args.accel_axis,
        accel_pitch_sign=args.accel_sign,
        gyro_pitch_axis=args.gyro_axis,
        gyro_pitch_sign=args.gyro_sign,
        gyro_units=args.gyro_units,
    )
    if cfg.loop_hz <= 0:
        parser.error("--loop-hz must be greater than 0")
    if cfg.output_limit > 30:
        parser.error("--output-limit is capped at 30 for this diagnostic script")
    return args, cfg


def main() -> int:
    args, cfg = parse_args()
    robot = connect(cfg)
    adapter = SdkAdapter(robot, cfg)

    if args.probe_sdk:
        run_capability_probe(adapter)

    if args.direction_test:
        require_not_builtin_balance(robot, args.force_custom_in_balance_mode)
        run_direction_test(adapter, cfg)

    needs_cal = args.calibrate or args.axis_test or args.run_custom_pid
    calibration: Optional[Calibration] = None
    if needs_cal:
        calibration = calibrate_stationary(adapter, cfg)

    if args.axis_test:
        run_axis_test(adapter, cfg, calibration)

    if args.run_custom_pid:
        require_not_builtin_balance(robot, args.force_custom_in_balance_mode)
        run_balance_loop(adapter, cfg, calibration, args.csv)

    if not any((args.probe_sdk, args.calibrate, args.axis_test, args.direction_test, args.run_custom_pid)):
        print("No action selected. Start with --probe-sdk, then --calibrate or --axis-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Interactive diagnostics for a UBTECH UGOT self-balancing car.

This script separates three failure classes:
    1. Robot/config problem: balance-only hold fails.
    2. Drive-command problem: balance-only works, but movement fails.
    3. Direction/assembly problem: one direction works but the other fails.

It still uses the UGOT firmware balance controller. The Python SDK does not
expose the balance PID gains, so do not try to tune PID here.
"""

import argparse
import sys
import time
from typing import Optional

from ugot import ugot


DEFAULT_IP = "192.168.1.77"
DEFAULT_HOLD_SECONDS = 12.0
DEFAULT_DRIVE_SECONDS = 5.0
DEFAULT_SPEED_CM_S = 8
DEFAULT_WARMUP_SECONDS = 3.0
COMMAND_REFRESH_SECONDS = 0.1
MIN_DRIVE_SPEED_CM_S = 5
MAX_DRIVE_SPEED_CM_S = 80


def connect(ip: str) -> ugot.UGOT:
    print(f"[INFO] Connecting to {ip} ...")
    robot = ugot.UGOT()
    robot.initialize(ip)
    return robot


def get_chassis_mode(robot: ugot.UGOT) -> Optional[str]:
    try:
        return robot.DEVICE.getDeviceModel()
    except Exception as exc:
        print(f"[WARN] Could not read chassis mode: {exc}")
        return None


def countdown(seconds: int, label: str) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"{label} in {remaining}s...", end="\r")
        time.sleep(1)
    print(" " * 60, end="\r")


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def stop_balance(robot: ugot.UGOT) -> None:
    try:
        robot.balance_stop_balancing()
    except Exception as exc:
        print(f"[WARN] Could not stop balance mode cleanly: {exc}")


def balance_hold_test(robot: ugot.UGOT, hold_seconds: float, warmup_seconds: float) -> bool:
    print("\n[TEST 1] Balance-only hold")
    print("[CHECK] Set chassis mode to Self-Balancing Car.")
    print("[CHECK] Confirm left wheel is motor port 1 and right wheel is motor port 3.")
    print("[CHECK] Use a flat floor, high battery, clean tires, and no payload hanging off-center.")
    input("[ACTION] Hold the robot upright, then press Enter to start balance mode.")

    countdown(int(warmup_seconds), "Starting balance")
    robot.balance_start_balancing()
    print("[ACTION] Let go carefully. No drive commands are being sent.")

    start = time.monotonic()
    while time.monotonic() - start < hold_seconds:
        elapsed = time.monotonic() - start
        print(f"        observing hold: {elapsed:4.1f}s / {hold_seconds:.0f}s", end="\r")
        time.sleep(0.2)
    print()

    passed = ask_yes_no("Did it remain upright without you holding it?")
    stop_balance(robot)
    return passed


def command_drive(robot: ugot.UGOT, speed_cm_s: int) -> None:
    direction = (
        ugot.E_Model.Direction.forward
        if speed_cm_s > 0
        else ugot.E_Model.Direction.backward
    )
    robot.balance_move_speed(direction, abs(speed_cm_s))


def drive_test(robot: ugot.UGOT, speed_cm_s: int, drive_seconds: float) -> bool:
    label = "forward" if speed_cm_s > 0 else "backward"
    print(f"\n[TEST] Balance drive {label} at {abs(speed_cm_s)} cm/s")
    input("[ACTION] Hold/support the robot upright again, then press Enter.")
    robot.balance_start_balancing()
    time.sleep(DEFAULT_WARMUP_SECONDS)
    print("[ACTION] Let go carefully. The robot will now move slowly.")

    start = time.monotonic()
    while time.monotonic() - start < drive_seconds:
        command_drive(robot, speed_cm_s)
        elapsed = time.monotonic() - start
        print(f"        driving {label}: {elapsed:4.1f}s / {drive_seconds:.0f}s", end="\r")
        time.sleep(COMMAND_REFRESH_SECONDS)
    print()

    passed = ask_yes_no(f"Did it stay upright while moving {label}?")
    stop_balance(robot)
    return passed


def print_diagnosis(hold_ok: bool, forward_ok: Optional[bool], backward_ok: Optional[bool]) -> None:
    print("\n[DIAGNOSIS]")
    if not hold_ok:
        print("Balance-only failed. This is not a movement PID issue in this code.")
        print("Most likely causes: wrong chassis mode, wrong motor ports, low battery, poor tire grip, bad center of mass, or IMU/calibration problem.")
        print("Fix balance-only first before testing drive commands.")
        return

    if forward_ok and backward_ok:
        print("Balance and slow movement both passed. Increase speed gradually only after this remains repeatable.")
        return

    if forward_ok != backward_ok:
        print("Only one direction passed. Check motor orientation, wheel mounting, center of mass bias, and whether forward/backward is reversed for this build.")
        return

    print("Balance-only passed, but movement failed. Keep using the balance API, lower speed, increase warmup, and test on a grippier flat surface.")
    print("If it falls immediately when movement starts, the robot may be overcorrecting because of tire slip, battery sag, or center-of-mass offset.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive UGOT self-balance diagnostics")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"Robot IP address (default: {DEFAULT_IP})")
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--drive-seconds", type=float, default=DEFAULT_DRIVE_SECONDS)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED_CM_S)
    parser.add_argument("--skip-drive", action="store_true", help="Only run the balance-hold test.")
    args = parser.parse_args()

    if args.hold_seconds <= 0:
        parser.error("--hold-seconds must be greater than 0")
    if args.drive_seconds <= 0:
        parser.error("--drive-seconds must be greater than 0")
    if not MIN_DRIVE_SPEED_CM_S <= abs(args.speed) <= MAX_DRIVE_SPEED_CM_S:
        parser.error(f"--speed magnitude must be {MIN_DRIVE_SPEED_CM_S}..{MAX_DRIVE_SPEED_CM_S} cm/s")

    robot = connect(args.ip)
    model = get_chassis_mode(robot)
    if model is not None:
        print(f"[INFO] Chassis mode reported by SDK: {model}")
        if model != "balance":
            print("[WARN] Expected chassis mode 'balance'. Change the robot/app mode before trusting test results.")

    try:
        hold_ok = balance_hold_test(robot, args.hold_seconds, DEFAULT_WARMUP_SECONDS)
        forward_ok = None
        backward_ok = None

        if hold_ok and not args.skip_drive:
            forward_ok = drive_test(robot, abs(args.speed), args.drive_seconds)
            backward_ok = drive_test(robot, -abs(args.speed), args.drive_seconds)

        print_diagnosis(hold_ok, forward_ok, backward_ok)
        return 0 if hold_ok else 2
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Stopping balance mode.")
        stop_balance(robot)
        return 130


if __name__ == "__main__":
    sys.exit(main())

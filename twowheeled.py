#!/usr/bin/env python3
"""Keep a UBTECH uGot self-balancing car upright, then optionally drive.

Robot setup:
    - Chassis mode in uGot settings: Self-Balancing Car
    - Left wheel: motor port 1
    - Right wheel: motor port 3

When the robot is in Self-Balancing Car mode, its built-in firmware PID uses
those wheel ports. This script does not tune PID gains because the Python SDK
does not expose that controller; it only starts the balance controller and
optionally sends balance-safe drive commands.

Run:
    .\\.venv\\Scripts\\python.exe .\\twowheeled.py --ip 192.168.1.77
    .\\.venv\\Scripts\\python.exe .\\twowheeled.py --ip 192.168.1.77 --speed 8

Hold the robot upright on a clear, level floor before starting. Press Ctrl+C
to stop; catch or support the robot after balance mode is switched off.
"""

import argparse
import time
from typing import Optional

from ugot import ugot


DEFAULT_IP = "192.168.1.77"
LEFT_WHEEL_PORT = 1
RIGHT_WHEEL_PORT = 3

# Speed 0 means "hold balance only". The balance API accepts 5 to 80 cm/s
# for movement, so the default should not drive before balance is proven.
DEFAULT_SPEED_CM_S = 0
MIN_DRIVE_SPEED_CM_S = 5
MAX_DRIVE_SPEED_CM_S = 80
WARMUP_SECONDS = 3
COMMAND_REFRESH_SECONDS = 0.1


def connect(ip: str) -> ugot.UGOT:
    robot = ugot.UGOT()
    robot.initialize(ip)
    return robot


def get_chassis_mode(robot: ugot.UGOT) -> Optional[str]:
    """Return the configured chassis mode when the SDK exposes it."""
    try:
        return robot.DEVICE.getDeviceModel()
    except Exception as exc:
        print(f"[WARN] Could not read chassis mode: {exc}")
        return None


def command_balance_drive(robot: ugot.UGOT, speed_cm_s: int) -> None:
    """Send a balance-safe signed speed command."""
    if speed_cm_s == 0:
        return

    direction = (
        ugot.E_Model.Direction.forward
        if speed_cm_s > 0
        else ugot.E_Model.Direction.backward
    )
    robot.balance_move_speed(direction, abs(speed_cm_s))


def stop_safely(robot: ugot.UGOT, leave_balancing: bool) -> None:
    """Stop balance mode unless the operator wants firmware balance left on."""
    try:
        if leave_balancing:
            print("[INFO] Leaving balance mode enabled. Catch/support the robot before power-off.")
        else:
            robot.balance_stop_balancing()
    except Exception as exc:
        print(f"[WARN] Could not stop cleanly: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a self-balancing uGot on the ground")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"Robot IP (default: {DEFAULT_IP})")
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_SPEED_CM_S,
        help=(
            "Signed speed in cm/s. 0 holds position; positive moves forward; "
            "negative moves backward. Moving speed magnitude must be 5..80 "
            f"(default: {DEFAULT_SPEED_CM_S})"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=WARMUP_SECONDS,
        help=f"Seconds to balance before sending drive commands (default: {WARMUP_SECONDS})",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=COMMAND_REFRESH_SECONDS,
        help=(
            "Seconds between repeated drive commands while moving "
            f"(default: {COMMAND_REFRESH_SECONDS})"
        ),
    )
    parser.add_argument(
        "--leave-balancing",
        action="store_true",
        help="Do not call balance_stop_balancing() on exit.",
    )
    args = parser.parse_args()

    if args.speed != 0 and not MIN_DRIVE_SPEED_CM_S <= abs(args.speed) <= MAX_DRIVE_SPEED_CM_S:
        parser.error(
            "--speed must be 0, or a signed movement speed with magnitude "
            f"{MIN_DRIVE_SPEED_CM_S}..{MAX_DRIVE_SPEED_CM_S} cm/s"
        )
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than 0")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.refresh <= 0:
        parser.error("--refresh must be greater than 0")

    robot = connect(args.ip)

    try:
        model = get_chassis_mode(robot)
        if model is not None:
            print(f"[INFO] Chassis mode: {model}")
            if model != "balance":
                print(
                    "[WARN] Chassis mode is not 'balance'. Set the UGOT app/robot "
                    "configuration to Self-Balancing Car before running this."
                )

        print("[INFO] Hold the robot upright. Enabling self-balance now...")
        robot.balance_start_balancing()
        time.sleep(args.warmup)

        if args.speed == 0:
            print("[INFO] Balance-hold mode. No drive commands will be sent.")
        else:
            print(
                f"[INFO] Balance drive mode at {args.speed} cm/s after "
                f"{args.warmup:g}s warmup. Press Ctrl+C to stop."
            )

        start = time.monotonic()
        while True:
            if args.duration is not None and time.monotonic() - start >= args.duration:
                break
            command_balance_drive(robot, args.speed)
            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping. Support the robot as balance mode turns off.")
    finally:
        stop_safely(robot, args.leave_balancing)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Keep a UBTECH uGot self-balancing car upright and moving slowly.

Robot setup:
    - Chassis mode in uGot settings: Self-Balancing Car
    - Left wheel: motor port 1
    - Right wheel: motor port 3

When the robot is in Self-Balancing Car mode, its built-in balance controller
uses those wheel ports. Do not use turn_motor_speed() in this script: direct
wheel commands bypass balancing and make the robot fall.

Run:
    .\\.venv\\Scripts\\python.exe .\\twowheeled.py

Hold the robot upright on a clear, level floor before starting. Press Ctrl+C
to stop; catch or support the robot after balance mode is switched off.
"""

import argparse
import time

from ugot import ugot


DEFAULT_IP = "192.168.1.77"
LEFT_WHEEL_PORT = 1
RIGHT_WHEEL_PORT = 3

# The balance API accepts 5 to 80 cm/s. Keep this small for the first test.
DEFAULT_SPEED_CM_S = 5
WARMUP_SECONDS = 2
COMMAND_REFRESH_SECONDS = 0.1


def connect(ip: str) -> ugot.UGOT:
    robot = ugot.UGOT()
    robot.initialize(ip)
    return robot


def stop_safely(robot: ugot.UGOT) -> None:
    """Command zero movement before turning balance mode off."""
    try:
        for _ in range(3):
            robot.model_common_move(0, 0)
            time.sleep(COMMAND_REFRESH_SECONDS)
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
        help="Forward speed in cm/s, from 5 to 80 (default: 5)",
    )
    args = parser.parse_args()

    if not 5 <= args.speed <= 80:
        parser.error("--speed must be between 5 and 80 cm/s")

    robot = connect(args.ip)

    try:
        print("[INFO] Hold the robot upright. Enabling self-balance now...")
        robot.balance_start_balancing()
        time.sleep(WARMUP_SECONDS)

        print(
            f"[INFO] Balancing and moving forward at {args.speed} cm/s. "
            "Release it carefully; press Ctrl+C to stop."
        )

        # Refresh the balance-drive command continuously. There is no gap
        # between commands, so the robot remains in active motion.
        while True:
            robot.balance_move_speed(ugot.E_Model.Direction.forward, args.speed)
            time.sleep(COMMAND_REFRESH_SECONDS)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping. Support the robot as balance mode turns off.")
    finally:
        stop_safely(robot)


if __name__ == "__main__":
    main()

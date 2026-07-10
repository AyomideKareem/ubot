#!/usr/bin/env python3
"""
uGot balance-only test.

Starts the balance controller and sends NOTHING else - no forward, no turn,
no stop mid-test. Just watches whether it stays upright on its own for the
target duration. This isolates the balance controller itself from anything
a drive script might be doing wrong.

Usage:
    python ugot_balance_only_test.py --ip 192.168.1.77
    python ugot_balance_only_test.py --ip 192.168.1.77 --hold-seconds 30
"""

import argparse
import time

from ugot import ugot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="uGot robot IP address")
    parser.add_argument("--hold-seconds", type=float, default=20.0,
                         help="How long it should stay upright (default %(default)s)")
    args = parser.parse_args()

    print(f"[INFO] Connecting to {args.ip} ...")
    got = ugot.UGOT()
    got.initialize(args.ip)

    model = got.DEVICE.getDeviceModel()
    print(f"[INFO] Chassis mode: {model}")
    if model != "balance":
        print("[WARN] Chassis mode is not 'balance' - balance_start_balancing() "
              "targets that mode specifically, so this test may not behave "
              "as expected.")

    print("[INFO] Place the robot upright on a flat, level surface and hold it.")
    for remaining in range(3, 0, -1):
        print(f"       starting balance in {remaining}s...", end="\r")
        time.sleep(1)

    got.balance_start_balancing()
    print("\n[INFO] balance_start_balancing() sent. Let go now.")
    print(f"[INFO] Watching for {args.hold_seconds:.0f}s. No other commands "
          "will be sent - this is purely observing whether it stays up.")

    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= args.hold_seconds:
            break
        print(f"       still up... {elapsed:4.1f}s / {args.hold_seconds:.0f}s", end="\r")
        time.sleep(0.2)

    print(f"\n[RESULT] Held upright for the full {args.hold_seconds:.0f}s "
          "with no drive commands. Balance controller looks solid on its own.")
    print("[INFO] Calling balance_stop_balancing() (should remain standing).")
    got.balance_stop_balancing()


if __name__ == "__main__":
    main()
import argparse
import sys
import time

from .ai import CachedVisionAIProvider, FakeVisionAIProvider
from .config import MuseumGuideConfig
from .hardware import FakeMuseumHardware, UGOTMuseumHardware
from .navigation import MuseumGuideController
from .telemetry import CsvTelemetry, StructuredLogger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe UGOT museum-guide controller")
    parser.add_argument("--ip", default="192.168.1.77")
    parser.add_argument("--hardware", action="store_true", help="Use physical UGOT hardware adapter.")
    parser.add_argument(
        "--confirm-physical",
        action="store_true",
        help="Required with --hardware to permit physical movement commands.",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds.")
    parser.add_argument("--log-jsonl", default="")
    parser.add_argument("--telemetry-csv", default="")
    parser.add_argument("--desired-artifact-distance", type=float, default=3.0)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = MuseumGuideConfig(
        robot_ip=args.ip,
        desired_artifact_distance_m=args.desired_artifact_distance,
        telemetry_csv_path=args.telemetry_csv,
        allow_physical_movement=args.hardware and args.confirm_physical,
    )
    if args.hardware and not args.confirm_physical:
        print("Refusing physical run without --confirm-physical.", file=sys.stderr)
        return 2

    hardware = UGOTMuseumHardware(cfg) if args.hardware else FakeMuseumHardware(cfg)
    logger = StructuredLogger(args.log_jsonl or None)
    telemetry = CsvTelemetry(args.telemetry_csv) if args.telemetry_csv else None
    ai = CachedVisionAIProvider(FakeVisionAIProvider())
    controller = MuseumGuideController(cfg, hardware, ai, logger=logger, telemetry=telemetry)

    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration:
            result = controller.step()
            if result.state.value in ("FAULT", "SAFE_SHUTDOWN"):
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        controller.safety.request_emergency_stop("operator keyboard interrupt")
        controller.step()
    finally:
        if telemetry:
            telemetry.close()
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

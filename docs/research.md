# UGOT Museum Guide Research and Implementation Plan

Date: 2026-07-11

## Repository Audit

Existing files:

- `twowheeled.py`: two-wheel self-balancing script. It initializes `ugot.UGOT()`, calls `initialize(ip)`, checks `DEVICE.getDeviceModel()`, starts `balance_start_balancing()`, and moves only through `balance_move_speed(...)`.
- `ugot_balance_diagnostics.py`: guided balance-hold and slow forward/backward diagnostics using UGOT built-in balance APIs.
- `ugot_balance_only_test.py`: older balance-only timer test using `balance_start_balancing()` and `balance_stop_balancing()`.
- `ugot_custom_balance_lab.py`: guarded custom-control lab for IMU/motor capability probing, calibration, pitch-axis checks, low-power motor sign tests, and custom PID experiments outside built-in balance mode.
- `ugot_wro_brick_driver.py`: mecanum/camera WRO driver. It uses `open_camera()`, `read_camera_data()`, `mecanum_move_xyz(...)`, and `mecanum_stop()`.
- `ugot_ackermann_fsm_driver.py`: Ackermann/camera driver. It uses `open_camera()`, `read_camera_data()`, direct motor/servo adapter methods, and an FSM for red/green obstacle handling.
- `requirements.txt`: dependencies include `ugot==0.2.13`, `opencv-python`, `numpy`, `grpcio`, `requests`, and supporting packages.

Balance audit:

- The production self-balancing path uses UGOT's built-in balance-car firmware controller.
- There is no local IMU read, pitch estimator, or custom PID in `twowheeled.py`.
- `ugot_custom_balance_lab.py` is explicitly guarded and refuses custom PID while the robot reports `balance` chassis mode unless forced.
- Direct motor commands appear only in non-balance drivers or the guarded lab.
- Slow camera/AI/navigation code is not currently inside the built-in balance loop.

Most likely balance root cause in this repository:

- Not PID constants in this repo. The code delegates balancing to UGOT firmware.
- Likely causes to verify physically: chassis mode not `balance`, wrong motor ports, battery sag, poor tire grip, center-of-mass offset, mechanical assembly, or internal UGOT IMU calibration.

## Research Sources

- PyPI `ugot 0.2.13`: https://pypi.org/project/ugot/
  - Confirms the installed SDK package name, latest pinned version used by this repo, release date May 14, 2024, Python >=3.7, and package dependencies.
- Local repository usage of SDK methods is currently the most concrete API evidence available:
  - `balance_start_balancing()`
  - `balance_stop_balancing()`
  - `balance_move_speed(direction, speed)`
  - `open_camera()`
  - `read_camera_data()`
  - `mecanum_move_xyz(x, y, z)`
  - `mecanum_stop()`
  - `DEVICE.getDeviceModel()`
- Public web search did not locate official UBTECH API pages for ultrasonic, TTS, IMU, or PID methods. Those APIs must remain behind adapters and be marked as requiring hardware SDK verification.

## SDK Constraints

- Do not stack a custom motor PID on top of UGOT built-in balance mode.
- The SDK balance PID gains are not documented or exposed in this repo.
- Camera capture via `open_camera()` and `read_camera_data()` is demonstrated by existing code.
- Mecanum movement APIs are demonstrated but are not appropriate for two-wheel self-balancing mode.
- Ultrasonic/distance, text-to-speech, IMU, and emergency-stop methods are not confirmed by official docs in this environment.
- Unconfirmed methods must be isolated in adapters with capability checks and safe failure behavior.

## Implementation Plan

1. Preserve existing balance scripts and unrelated WRO/Ackermann behavior.
2. Add a new `museum_guide` package with separate modules for configuration, data models, hardware adapters, balance/motion facade, sensors, perception, navigation FSM, artifact tracking, AI provider interface, speech, safety, telemetry, and simulation.
3. Make the default runnable mode simulation/fake hardware only. Physical movement must require an explicit `--hardware` plus `--confirm-physical` style operator confirmation.
4. Use UGOT built-in balance APIs for the physical adapter. Keep `balance_start_balancing()` active throughout normal operation; use `balance_move_speed(...)` for forward/reverse self-balancing motion where confirmed by repo usage.
5. Do not invent ultrasonic/TTS API calls. Provide adapter capability placeholders that fail safe when unavailable.
6. Implement museum navigation as a testable state machine: startup, calibration gate, patrolling, artifact candidate, approaching, positioning, capturing, identifying, presenting, backing away, turning, recovering, safe shutdown, fault.
7. Use distance readings only for collision zones and approximate range. Use camera perception and multi-frame confirmation for artifact candidates. Never classify people as artifacts.
8. Run AI and speech through asynchronous-style job queues in the controller design; tests will step the controller synchronously with fake completion events.
9. Add structured JSONL logging and optional CSV telemetry hooks at reduced rates.
10. Add fake-hardware and unit tests for navigation, safety, artifact confirmation, AI/speech failures, stale sensors, emergency stop, excessive tilt, and balance-not-disabled guarantees.
11. Add an example environment file without secrets.
12. After implementation, run available local checks. If Python is unavailable on PATH, record that as a tooling blocker and still run static checks available through git.

## Hardware Test Gate

Before autonomous museum navigation on the physical robot:

1. Run `python ugot_balance_diagnostics.py --ip <robot-ip> --skip-drive`.
2. Confirm chassis mode reports `balance`.
3. Confirm balance-only hold is stable.
4. Run slow forward/backward tests at low speed.
5. Probe SDK capabilities for distance/TTS/camera methods in a non-moving mode.
6. Only then run the museum guide with physical movement enabled, one operator ready to catch/disable the robot, in an empty controlled area.

# uBot

Python drivers and utilities for controlling a UBTECH uGot robot. This project includes self-balancing controllers, WRO-style obstacle navigation, and Ackermann steering implementations.

## Project Overview

This project provides multiple driver scripts for different robot configurations and use cases:
- **Self-balancing two-wheeled mode** for stable movement
- **Four-wheeled Ackermann steering** for WRO competitions with FSM-based obstacle avoidance
- **Mecanum drive with vision-based brick detection** for WRO Future Engineers tasks

All scripts communicate with the uGot robot over Wi-Fi.

## What is included

- `twowheeled.py` - Self-balancing car controller. Maintains upright balance while moving forward/backward.
- `ugot_balance_only_test.py` - Test script for balance mode without movement.
- `ugot_balance_diagnostics.py` - Interactive diagnostic sequence for balance-hold and slow forward/backward movement.
- `ugot_custom_balance_lab.py` - Guarded custom-control lab for SDK capability probing, IMU calibration, pitch-axis checks, low-power motor sign tests, and custom PID experiments when not using UGOT's built-in balance mode.
- `museum_guide/` - Safe museum-guide architecture with fake-hardware simulation, state machine, safety supervisor, artifact tracking, AI schema, speech queue, and physical UGOT adapter.
- `tests/` - Unit/simulation tests for the museum-guide safety and navigation logic.
- `docs/` - Research notes, SDK constraints, architecture, and safe physical test procedures.
- `ugot_ackermann_fsm_driver.py` - Ackermann steering controller with finite state machine for obstacle-aware navigation. Detects and avoids red/green bricks.
- `ugot_wro_brick_driver.py` - Brick-detection driver for WRO-style obstacle handling using mecanum wheels. Includes HSV-based color detection.
- `requirements.txt` - Python dependencies for the project.

## Requirements

- Python 3.7+
- UBTECH uGot robot with Wi-Fi connectivity
- Laptop or single-board computer on the same network as the robot

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/macOS
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the robot is powered on and connected to your Wi-Fi network.

## Usage

### Self-Balancing Mode
Run the guided diagnostic first:
```bash
python ugot_balance_diagnostics.py --ip 192.168.1.77
```

For direct balance-hold testing:
```bash
python twowheeled.py --ip 192.168.1.77
```
This starts the built-in balance controller without movement commands. Once that
holds reliably, test slow movement:
```bash
python twowheeled.py --ip 192.168.1.77 --speed 8 --duration 10
python twowheeled.py --ip 192.168.1.77 --speed -8 --duration 10
```
**Note:** Hold the robot upright on a clear, level floor before starting. Press Ctrl+C to stop.

### Custom Balance-Control Lab
Only use this if you are intentionally testing a custom controller outside the
UGOT built-in balance-car mode. Start by checking what the installed SDK exposes:
```bash
python ugot_custom_balance_lab.py --ip 192.168.1.77 --probe-sdk
```

If compatible IMU methods exist, run stationary calibration and pitch-axis checks:
```bash
python ugot_custom_balance_lab.py --ip 192.168.1.77 --calibrate
python ugot_custom_balance_lab.py --ip 192.168.1.77 --axis-test
```

If compatible direct motor methods exist, run only low-power sign tests with the
wheels safely supported:
```bash
python ugot_custom_balance_lab.py --ip 192.168.1.77 --direction-test
```

The custom PID loop is guarded and refuses to run while the robot reports
chassis mode `balance`, because that would stack a laptop PID on top of UGOT's
firmware balance controller.

### Museum Guide Simulation
Run the museum-guide stack with fake hardware first:
```bash
python -m museum_guide.runner --duration 10
```

Physical movement is guarded and requires explicit confirmation:
```bash
python -m museum_guide.runner --hardware --confirm-physical --ip 192.168.1.77 --duration 10
```

Do not run physical museum navigation until balance diagnostics pass and the
installed SDK methods for distance sensing, camera, balance-mode turning, and
text-to-speech have been verified. See `docs/research.md` and
`docs/museum_guide.md`.

### Ackermann Steering with FSM Navigation
```bash
python ugot_ackermann_fsm_driver.py --ip 10.196.72.185 --direction CCW
```
- Uses finite state machine to handle lane following and obstacle avoidance
- `--direction` can be `CW` (clockwise) or `CCW` (counterclockwise)

### WRO Brick Detection Driver
```bash
python ugot_wro_brick_driver.py --ip 10.196.72.185
```

For HSV color range tuning:
```bash
python ugot_wro_brick_driver.py --tune
```

## Configuration

- Update the `DEFAULT_IP` constant in each script if needed (default varies by script)
- Adjust motor/servo port settings based on your hardware configuration
- The brick driver includes HSV-based color detection that can be tuned for different lighting conditions

## Hardware Setup

### Two-Wheeled Self-Balancing
- Chassis mode: Self-Balancing Car
- Left wheel: motor port 1
- Right wheel: motor port 3

### Four-Wheeled Ackermann Steering
- Rear drive motors
- Front Ackermann steering actuator
- Refer to script comments for specific port configurations

## Notes

- These scripts are intended for a laptop or single-board computer connected to the robot over Wi-Fi
- The self-balancing script uses the UGOT firmware balance controller, not a local PID loop; the Python SDK does not expose balance PID gains
- Do not use direct wheel commands when in balance mode; use the balance API instead
- If balance-only mode fails, check chassis mode, wheel ports, battery level, tire grip, center of mass, and IMU/calibration before tuning movement behavior
- Do not run `ugot_custom_balance_lab.py --run-custom-pid` in UGOT `balance` chassis mode unless you deliberately force it and accept the risk
- The brick driver includes a simple HSV-based detection baseline that can be tuned further
- Press Ctrl+C to safely stop any running driver

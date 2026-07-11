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
- The brick driver includes a simple HSV-based detection baseline that can be tuned further
- Press Ctrl+C to safely stop any running driver

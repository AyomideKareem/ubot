# uBot

This project contains Python drivers for controlling a UBTECH uGot robot.

## What is included

- `ugot_ackermann_fsm_driver.py` - Ackermann steering controller for obstacle-aware navigation.
- `ugot_wro_brick_driver.py` - Brick-detection driver for WRO-style obstacle handling.
- `requirements.txt` - Python dependencies for the project.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the robot is on the same Wi-Fi network and update the robot IP address in the script if needed.

## Run the drivers

### Ackermann steering driver
```bash
python ugot_ackermann_fsm_driver.py --ip 10.196.72.185 --direction CCW
```

### WRO brick driver
```bash
python ugot_wro_brick_driver.py --ip 10.196.72.185
```

## Notes

- These scripts are intended for a laptop or single-board computer connected to the robot.
- You may need to adjust motor/servo port settings depending on your hardware setup.
- The brick driver includes a simple HSV-based detection baseline that can be tuned further.

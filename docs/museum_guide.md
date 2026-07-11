# Museum Guide System

## Architecture

The museum guide is implemented as a new `museum_guide` package so existing WRO,
Ackermann, and balance diagnostics remain unchanged.

- `config.py`: all tunable speeds, distances, timeouts, thresholds, privacy, and logging settings.
- `models.py`: state, safety, perception, motion, distance, and AI result schemas.
- `hardware.py`: physical UGOT adapter and fake-hardware adapter. Physical distance, TTS, and balance-turn methods are capability-checked because official APIs were not confirmed.
- `safety.py`: emergency stop, stale sensor, person-distance, excessive-tilt, balance-active, and obstacle-zone checks.
- `artifacts.py`: multi-frame camera-based artifact candidate tracker; people clear the candidate tracker.
- `ai.py`: provider-independent structured AI interface plus fake/cached provider.
- `local_catalog.py`: local reference-image matcher that maps captured images to trusted catalogue metadata.
- `speech.py`: non-blocking speech queue facade; failures are counted and do not disable balance.
- `telemetry.py`: JSONL structured logs and optional CSV telemetry.
- `navigation.py`: explicit state machine for patrol, artifact approach, capture, identification, presentation, back-away, turn, recovery, shutdown, and fault.
- `runner.py`: CLI entry point. Defaults to fake hardware. Physical runs require `--hardware --confirm-physical`.

## Safety Model

- Built-in UGOT balance mode is the normal physical balancing path.
- The physical adapter calls `balance_start_balancing()` and never calls `balance_stop_balancing()` during ordinary navigation.
- Ordinary stop/hold decisions stop translation only and keep balance active.
- Emergency stop, excessive tilt, communication failure, stale sensors, or human danger override normal behavior.
- Ultrasonic/distance readings are used only for collision zones and approximate range. They are never used alone to decide that something is an artifact.
- Artifact candidates require camera detections across multiple frames, confidence above threshold, stable target identity/region, and no person detection.
- People are never classified as artifacts. If a person appears, the robot yields/back-aways or faults safe.

## Configuration

Key fields in `MuseumGuideConfig`:

- `patrol_speed_cm_s`, `caution_speed_cm_s`, `reverse_speed_cm_s`, `turn_speed`
- `clear_distance_m`, `caution_distance_m`, `danger_distance_m`
- `human_safe_distance_m`
- `desired_artifact_distance_m` defaults to 3.0 m, but this must be verified against the installed sensor/camera range before physical use.
- `sensor_timeout_s`, `camera_timeout_s`, `movement_command_timeout_s`
- `confirmation_frames`
- `ai_confidence_threshold`, `artifact_confidence_threshold`
- `ai_timeout_s`
- `speech_timeout_s`
- `capture_sharpness_threshold`, `capture_brightness_threshold`
- `max_capture_attempts`, `max_recovery_attempts`
- `state_timeouts_s`
- `privacy_blur_people`, `retain_images`

## Local Catalogue Recognition

If you do not have a cloud vision model, use the local catalogue provider. This
does not train a new neural network. It builds an image-matching index from your
reference photos each time the program starts.

Create this layout:

```text
museum_catalog/
  artifacts.json
  images/
    bronze_mask/
      front.jpg
      side.jpg
      close.jpg
```

Each entry in `artifacts.json` should contain:

- `id`: stable unique ID, for example `bronze_mask_001`
- `name`: artifact name to speak
- `category`: broad category such as `Sculpture`
- `short_description`: concise factual visitor description
- `visible_evidence`: visible features that support the match
- `reference_images`: paths relative to `museum_catalog/`

Start from `museum_catalog/artifacts.example.json`, save your real file as
`museum_catalog/artifacts.json`, then run:

```bash
python -m museum_guide.runner --vision-provider local --catalog museum_catalog/artifacts.json --catalog-threshold 0.68 --duration 10
```

The matcher returns `unknown` when the best match is below the configured
threshold. The default is `0.68`, which is about 68 percent. Use
`--catalog-threshold` to tune this; raise it if false matches happen, lower it
only if correct matches are being missed.

## Safe Physical Test Procedure

1. Confirm the robot is in Self-Balancing Car mode with left wheel on port 1 and right wheel on port 3.
2. Run balance-only diagnostics: `python ugot_balance_diagnostics.py --ip <robot-ip> --skip-drive`.
3. Run slow balance movement diagnostics: `python ugot_balance_diagnostics.py --ip <robot-ip>`.
4. Verify SDK capabilities without movement using the museum guide in fake mode first:
   `python -m museum_guide.runner --duration 5`.
5. Confirm physical adapter methods for distance, camera, balance turn, and TTS on your robot. Unverified methods fail safe.
6. In an empty controlled area, with an operator ready to catch/disable the robot:
   `python -m museum_guide.runner --hardware --confirm-physical --ip <robot-ip> --duration 10`.
7. Do not use a museum environment with visitors until person detection and distance sensing are validated on the actual robot.

## Known Hardware Limitations

- Official documentation for ultrasonic, TTS, IMU, balance-turn, and emergency-stop methods was not confirmed in this environment.
- The 3.0 m viewing distance is configurable and software-supported, but sensor capability at that range is NOT VERIFIED.
- Camera-based perception in the current implementation is interface-driven; production object/person detection requires a real detector wired into `PerceptionFrame`.
- The physical UGOT adapter currently returns raw camera image bytes only. A real detector must convert camera frames into `VisionDetection` entries before artifact/person behavior is physically functional.
- Physical behavior remains NOT VERIFIED until tested on the UGOT.

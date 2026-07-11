# Local Museum Catalogue

This folder is the local "model" used by the robot.

It is not trained like a neural network. Instead, the robot builds a local image
index from reference photos and links each match to trusted metadata in
`artifacts.json`.

## Folder Layout

```text
museum_catalog/
  artifacts.json
  images/
    artifact_001/
      front.jpg
      left.jpg
      close.jpg
    artifact_002/
      front.jpg
      side.jpg
```

Use `artifacts.example.json` as a template, then rename your real file to
`artifacts.json`.

## How To Add An Artifact

1. Create a folder under `images/`, for example `images/bronze_mask/`.
2. Add 3-10 clear reference photos of the same artifact from likely robot
   viewing angles.
3. Add one object to `artifacts.json`.
4. Keep the description factual. The robot will speak this text only after a
   confident local image match.
5. Do not add photos of visitors or faces.

## Run With Local Catalogue

```bash
python -m museum_guide.runner --vision-provider local --catalog museum_catalog/artifacts.json --duration 10
```

Physical mode still requires the normal safety confirmation:

```bash
python -m museum_guide.runner --hardware --confirm-physical --vision-provider local --catalog museum_catalog/artifacts.json --ip 192.168.1.77 --duration 10
```


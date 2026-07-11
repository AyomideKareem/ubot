import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import cv2
import numpy as np

from .models import AIResult, PerceptionFrame


HASH_SIZE = 16
HASH_BITS = HASH_SIZE * HASH_SIZE


@dataclass
class ArtifactRecord:
    artifact_id: str
    name: str
    category: str
    short_description: str
    reference_images: List[str]
    visible_evidence: List[str]


@dataclass
class ReferenceSignature:
    artifact: ArtifactRecord
    image_path: str
    average_hash: int
    color_signature: Tuple[float, float, float]


class LocalCatalogVisionProvider:
    """Local image matcher backed by reference photos and trusted metadata."""

    def __init__(self, catalog_path: str, min_confidence: float = 0.86) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_dir = self.catalog_path.parent
        self.min_confidence = min_confidence
        self.references: List[ReferenceSignature] = []
        self.skipped_reference_images: List[str] = []
        self._load_catalog()

    def identify(self, frame: PerceptionFrame) -> AIResult:
        if frame.people():
            return AIResult(
                candidate_name="person",
                category="person",
                confidence=1.0,
                visible_evidence=["person detected in frame"],
                short_description="I will not identify visitors as museum artifacts.",
                uncertainty="The target region contains a person.",
                needs_human_review=False,
                safety_or_privacy_flags=["person_detected"],
            )
        if frame.image is None:
            return AIResult.uncertain("No image was available for local catalogue matching.")
        if not self.references:
            return AIResult.uncertain("The local catalogue contains no usable reference images.")

        try:
            image = load_image(frame.image)
        except Exception as exc:
            return AIResult.uncertain("Could not decode captured image: %s" % exc)

        query_hash = average_hash(image)
        query_color = color_signature(image)
        best_reference = None
        best_score = 0.0
        for reference in self.references:
            score = signature_similarity(
                query_hash,
                query_color,
                reference.average_hash,
                reference.color_signature,
            )
            if score > best_score:
                best_score = score
                best_reference = reference

        if best_reference is None or best_score < self.min_confidence:
            return AIResult(
                candidate_name="unknown",
                category="unknown",
                confidence=best_score,
                visible_evidence=["No local reference image matched above the configured threshold."],
                short_description="I am not sure which catalogue item this is.",
                uncertainty="Best local catalogue match was below threshold %.2f." % self.min_confidence,
                needs_human_review=True,
                safety_or_privacy_flags=[],
            )

        artifact = best_reference.artifact
        result = AIResult(
            candidate_name=artifact.name,
            category=artifact.category,
            confidence=best_score,
            visible_evidence=[
                "matched local reference image %s" % best_reference.image_path,
            ]
            + artifact.visible_evidence,
            short_description=artifact.short_description,
            uncertainty=(
                "Local visual match confidence %.2f. This is not proof of identity; "
                "verify with the museum catalogue when available."
            )
            % best_score,
            needs_human_review=best_score < 0.95,
            safety_or_privacy_flags=[],
        )
        result.validate()
        return result

    def _load_catalog(self) -> None:
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        records = data["artifacts"] if isinstance(data, dict) else data
        for raw in records:
            artifact = ArtifactRecord(
                artifact_id=str(raw["id"]),
                name=str(raw["name"]),
                category=str(raw.get("category", "museum artifact")),
                short_description=str(raw["short_description"]),
                reference_images=string_list(raw.get("reference_images", []), "reference_images"),
                visible_evidence=string_list(raw.get("visible_evidence", []), "visible_evidence"),
            )
            for image_path in artifact.reference_images:
                full_path = self.catalog_dir / image_path
                try:
                    image = load_image(full_path)
                except Exception:
                    self.skipped_reference_images.append(image_path)
                    continue
                self.references.append(
                    ReferenceSignature(
                        artifact=artifact,
                        image_path=image_path,
                        average_hash=average_hash(image),
                        color_signature=color_signature(image),
                    )
                )


def load_image(source: Any) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return ensure_bgr(source)
    if source.__class__.__module__.startswith("PIL."):
        array = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image from %s" % source)
        return image
    if isinstance(source, bytes):
        buffer = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes")
        return image
    if hasattr(source, "read"):
        buffer = np.frombuffer(source.read(), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image stream")
        return image
    if hasattr(source, "__array__"):
        array = np.asarray(source)
        if array.ndim == 3 and array.shape[2] >= 3:
            return cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
        return ensure_bgr(array)
    raise TypeError("Unsupported image source type: %s" % type(source).__name__)


def string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("%s must be a list of strings" % field_name)
    return value


def average_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    pixels = resized.flatten()
    avg = float(pixels.mean())
    value = 0
    for pixel in pixels:
        value = (value << 1) | (1 if float(pixel) >= avg else 0)
    return value


def color_signature(image: np.ndarray) -> Tuple[float, float, float]:
    bgr = ensure_bgr(image)
    resized = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
    mean_bgr = cv2.mean(resized)[:3]
    rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    return tuple(float(channel) for channel in rgb)


def signature_similarity(
    query_hash: int,
    query_color: Tuple[float, float, float],
    ref_hash: int,
    ref_color: Tuple[float, float, float],
) -> float:
    hash_score = 1.0 - hamming_distance(query_hash, ref_hash) / float(HASH_BITS)
    color_score = 1.0 - min(1.0, color_distance(query_color, ref_color) / 255.0)
    return max(0.0, min(1.0, 0.55 * hash_score + 0.45 * color_score))


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def color_distance(a: Iterable[float], b: Iterable[float]) -> float:
    pairs = list(zip(a, b))
    return sum(abs(x - y) for x, y in pairs) / float(len(pairs))


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

"""Data models for rebar detection results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    """Convert model values into JSON-compatible Python values."""
    if is_dataclass(value):
        return {field: _json_safe(item) for field, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_degrees: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class RebarLine:
    id: str
    direction_id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    center: tuple[float, float]
    angle_degrees: float
    normal_position: float
    support_segments: int
    support_length: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SpacingMeasurement:
    from_rebar_id: str
    to_rebar_id: str
    pixels: float
    value: float
    unit: str
    relative_deviation: float
    is_anomaly: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class DirectionResult:
    id: str
    name: str
    angle_degrees: float
    rebars: list[RebarLine]
    spacings: list[SpacingMeasurement]
    statistics: dict[str, float | int | str | None]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class DetectionReport:
    input_path: str
    output_path: str
    image_width: int
    image_height: int
    rectified: bool
    mm_per_pixel: float | None
    parameters: dict
    directions: list[DirectionResult]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

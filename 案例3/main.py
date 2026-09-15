"""单文件版钢筋检测程序。原始模块副本保留在 rebar_detector/ 中。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import yaml

# 直接修改下面的参数后运行 `python main.py`。
PROJECT_DIR = Path(__file__).resolve().parent
# 可选输入图片路径（当前只保留这 5 张图片）
SELECTED_DATASET_IMAGES = [
    PROJECT_DIR / "Dataset" / "rebar_01.jpg",
    PROJECT_DIR / "Dataset" / "rebar_02.jpg",
    PROJECT_DIR / "Dataset" / "rebar_03.jpg",
    PROJECT_DIR / "Dataset" / "rebar_04.jpg",
    PROJECT_DIR / "Dataset" / "rebar_05.jpg",
]
# 输入图片：0，1，2，3，4
INPUT_IMAGE_PATH = SELECTED_DATASET_IMAGES[0]
# 输出目录
OUTPUT_DIR = PROJECT_DIR / "output"
# 没有物理标定时保持 None
MM_PER_PIXEL = None
# 透视点格式："左上x,左上y;右上x,右上y;右下x,右下y;左下x,左下y"
PERSPECTIVE_POINTS = None


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

_DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "default.yaml"

_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "preprocess": (
        "gaussian_kernel",
        "canny_low",
        "canny_high",
        "morphology_enabled",
        "morphology_kernel",
    ),
    "hough": (
        "rho",
        "theta_degrees",
        "threshold",
        "min_line_length",
        "max_line_gap",
    ),
    "grouping": (
        "angle_tolerance_degrees",
        "max_direction_groups",
        "min_group_segments",
        "min_group_total_length",
    ),
    "merging": (
        "normal_cluster_distance_px",
        "duplicate_center_distance_px",
        "min_support_segments",
        "min_support_length",
        "min_axis_coverage_px",
    ),
    "measurement": ("anomaly_ratio", "low_sample_spacing_count"),
    "output": ("save_debug", "draw_raw_segments"),
}

def _require_number(section: Mapping[str, Any], key: str, *, integer: bool = False) -> float | int:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value

def _require_positive(section: Mapping[str, Any], key: str, *, integer: bool = False) -> None:
    if _require_number(section, key, integer=integer) <= 0:
        raise ValueError(f"{key} must be positive")

def validate_config(config: Mapping[str, Any]) -> None:
    """Validate all required detector settings.

    A ``ValueError`` is raised for missing sections, missing keys, invalid types,
    or values outside the supported detector ranges.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    sections: dict[str, Mapping[str, Any]] = {}
    for section_name, keys in _REQUIRED_KEYS.items():
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"missing mapping: {section_name}")
        missing = [key for key in keys if key not in section]
        if missing:
            raise ValueError(f"missing {section_name} keys: {', '.join(missing)}")
        sections[section_name] = section

    preprocess = sections["preprocess"]
    gaussian_kernel = _require_number(preprocess, "gaussian_kernel", integer=True)
    if gaussian_kernel <= 0 or gaussian_kernel % 2 == 0:
        raise ValueError("gaussian_kernel must be a positive odd integer")
    morphology_kernel = _require_number(preprocess, "morphology_kernel", integer=True)
    if morphology_kernel <= 0 or morphology_kernel % 2 == 0:
        raise ValueError("morphology_kernel must be a positive odd integer")
    morphology_enabled = preprocess["morphology_enabled"]
    if not isinstance(morphology_enabled, bool):
        raise ValueError("morphology_enabled must be boolean")
    canny_low = _require_number(preprocess, "canny_low")
    canny_high = _require_number(preprocess, "canny_high")
    if not 0 <= canny_low < canny_high:
        raise ValueError("canny thresholds must satisfy 0 <= canny_low < canny_high")

    hough = sections["hough"]
    for key in _REQUIRED_KEYS["hough"]:
        _require_positive(hough, key)

    grouping = sections["grouping"]
    _require_positive(grouping, "angle_tolerance_degrees")
    max_groups = _require_number(grouping, "max_direction_groups", integer=True)
    if not 1 <= max_groups <= 2:
        raise ValueError("max_direction_groups must be between 1 and 2")
    _require_positive(grouping, "min_group_segments", integer=True)
    _require_positive(grouping, "min_group_total_length")

    merging = sections["merging"]
    for key in _REQUIRED_KEYS["merging"]:
        _require_positive(merging, key, integer=key == "min_support_segments")

    measurement = sections["measurement"]
    anomaly_ratio = _require_number(measurement, "anomaly_ratio")
    if not 0 < anomaly_ratio < 1:
        raise ValueError("anomaly_ratio must satisfy 0 < anomaly_ratio < 1")
    _require_number(measurement, "low_sample_spacing_count", integer=True)
    if measurement["low_sample_spacing_count"] < 0:
        raise ValueError("low_sample_spacing_count must not be negative")

    output = sections["output"]
    for key in _REQUIRED_KEYS["output"]:
        if not isinstance(output[key], bool):
            raise ValueError(f"{key} must be boolean")

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load, validate, and return a UTF-8 YAML detector configuration."""
    config_path = _DEFAULT_CONFIG_PATH if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(config)
    return config

__all__ = [
    "clip_infinite_line_to_image",
    "direction_vector",
    "normal_vector",
    "normalize_angle_degrees",
    "project_point",
    "segment_from_points",
    "undirected_angle_distance",
]

def normalize_angle_degrees(angle: float) -> float:
    """Normalize an undirected line angle to the half-open range [0, 180)."""
    return float(angle % 180.0)

def undirected_angle_distance(a: float, b: float) -> float:
    """Return the smallest angular distance between two undirected lines."""
    raw = abs(normalize_angle_degrees(a) - normalize_angle_degrees(b))
    return min(raw, 180.0 - raw)

def segment_from_points(points: tuple[float, float, float, float]) -> LineSegment:
    """Build a :class:`LineSegment` from two endpoint coordinates."""
    x1, y1, x2, y2 = map(float, points)
    dx, dy = x2 - x1, y2 - y1
    return LineSegment(
        x1,
        y1,
        x2,
        y2,
        math.hypot(dx, dy),
        normalize_angle_degrees(math.degrees(math.atan2(dy, dx))),
    )

def direction_vector(angle: float) -> tuple[float, float]:
    """Return the unit vector for an angle measured counter-clockwise from x."""
    radians = math.radians(angle)
    return (math.cos(radians), math.sin(radians))

def normal_vector(angle: float) -> tuple[float, float]:
    """Return the left-hand unit normal for an angle measured from x."""
    radians = math.radians(angle)
    return (-math.sin(radians), math.cos(radians))

def project_point(point: Sequence[float], axis: Sequence[float]) -> float:
    """Project a point onto a supplied axis using a dot product.

    The caller supplies the axis, which is expected to be a unit vector.  The
    axis is intentionally not normalized here so the helper preserves the
    supplied projection convention.
    """
    x, y = map(float, point)
    axis_x, axis_y = map(float, axis)
    return float(x * axis_x + y * axis_y)

def clip_infinite_line_to_image(
    center: Sequence[float],
    direction: Sequence[float],
    width: int | float,
    height: int | float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Clip an infinite parametric line to an image rectangle.

    The rectangle uses pixel-coordinate bounds ``0 .. width - 1`` and
    ``0 .. height - 1``.  Intersections with all four boundaries are found,
    duplicate corner intersections are removed, and the farthest pair is
    returned.  A line with no valid extent through the image raises
    ``ValueError``.
    """
    cx, cy = map(float, center)
    dx, dy = map(float, direction)
    image_width = float(width)
    image_height = float(height)

    if dx == 0.0 and dy == 0.0:
        raise ValueError("direction must be non-zero")
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image dimensions must be positive")

    xmax = image_width - 1.0
    ymax = image_height - 1.0
    epsilon = 1e-9
    intersections: list[tuple[float, float]] = []

    def add_if_in_bounds(point: tuple[float, float]) -> None:
        x, y = point
        if -epsilon <= x <= xmax + epsilon and -epsilon <= y <= ymax + epsilon:
            bounded = (
                min(max(x, 0.0), xmax),
                min(max(y, 0.0), ymax),
            )
            if not any(
                math.isclose(bounded[0], existing[0], abs_tol=epsilon)
                and math.isclose(bounded[1], existing[1], abs_tol=epsilon)
                for existing in intersections
            ):
                intersections.append(bounded)

    if dx != 0.0:
        for x in (0.0, xmax):
            t = (x - cx) / dx
            add_if_in_bounds((x, cy + t * dy))

    if dy != 0.0:
        for y in (0.0, ymax):
            t = (y - cy) / dy
            add_if_in_bounds((cx + t * dx, y))

    if len(intersections) < 2:
        raise ValueError("line has fewer than two image intersections")

    farthest_pair = max(
        (
            (intersections[index], intersections[other_index])
            for index in range(len(intersections))
            for other_index in range(index + 1, len(intersections))
        ),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2
        + (pair[0][1] - pair[1][1]) ** 2,
    )
    return farthest_pair

@dataclass
class PreprocessResult:
    """Images produced by the preprocessing pipeline."""

    working: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    rectified: bool

def parse_perspective_points(text: str | None) -> list[tuple[float, float]] | None:
    """Parse four ``x,y`` perspective points separated by semicolons.

    The returned points retain the caller's required order: top-left,
    top-right, bottom-right, bottom-left.  ``None`` and blank text mean that
    no perspective correction was requested.
    """
    if text is None or not text.strip():
        return None

    raw_points = [part.strip() for part in text.split(";")]
    if len(raw_points) != 4:
        raise ValueError("透视矫正必须提供四个点")

    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        coordinates = [part.strip() for part in raw_point.split(",")]
        if len(coordinates) != 2 or not all(coordinates):
            raise ValueError("透视点必须使用 x,y 格式")
        try:
            x, y = (float(value) for value in coordinates)
        except ValueError as exc:
            raise ValueError("透视点坐标必须是数字") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("透视点坐标必须是有限数字")
        points.append((x, y))
    return points

def _as_perspective_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    if len(points) != 4:
        raise ValueError("透视矫正必须提供四个点")
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("透视点坐标必须是数字") from exc
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("透视点坐标必须是有限的二维坐标")
    return values

def _cross_product(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])

def _segments_properly_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    first_side_a = _cross_product(first_start, first_end, second_start)
    first_side_b = _cross_product(first_start, first_end, second_end)
    second_side_a = _cross_product(second_start, second_end, first_start)
    second_side_b = _cross_product(second_start, second_end, first_end)
    first_straddles = (first_side_a > tolerance and first_side_b < -tolerance) or (
        first_side_a < -tolerance and first_side_b > tolerance
    )
    second_straddles = (
        second_side_a > tolerance and second_side_b < -tolerance
    ) or (second_side_a < -tolerance and second_side_b > tolerance)
    return first_straddles and second_straddles

def _validate_perspective_quadrilateral(points: np.ndarray) -> None:
    span = np.ptp(points, axis=0)
    scale = max(float(np.linalg.norm(span)), 1.0)
    distance_tolerance = scale * 1e-9
    area_tolerance = max(1e-6, scale * scale * 1e-8)

    for index in range(4):
        for other_index in range(index + 1, 4):
            if np.linalg.norm(points[index] - points[other_index]) <= distance_tolerance:
                raise ValueError("透视四边形不能包含重复点")

    if _segments_properly_intersect(
        points[0], points[1], points[2], points[3], area_tolerance
    ) or _segments_properly_intersect(
        points[1], points[2], points[3], points[0], area_tolerance
    ):
        raise ValueError("透视四边形的边不能相交")

    for index in range(4):
        previous_point = points[(index - 1) % 4]
        point = points[index]
        next_point = points[(index + 1) % 4]
        if abs(_cross_product(previous_point, point, next_point)) <= area_tolerance:
            raise ValueError("透视四边形存在共线点或退化边")

    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    twice_area = float(
        np.dot(x_coordinates, np.roll(y_coordinates, -1))
        - np.dot(y_coordinates, np.roll(x_coordinates, -1))
    )
    if abs(twice_area) / 2.0 <= area_tolerance:
        raise ValueError("透视四边形面积过小或已经退化")

def _validate_numeric_array(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")

def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a two- or three-dimensional array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")
    _validate_numeric_array(image, "image")

def rectify_perspective(image: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Warp an image from four ordered corners onto a rectangle.

    Target dimensions are the rounded maximum lengths of the opposing
    source sides.  Pixel bounds use ``width - 1`` and ``height - 1`` so that
    an axis-aligned rectangle retains the expected dimensions.
    """
    _validate_image(image)
    source = _as_perspective_array(points)
    _validate_perspective_quadrilateral(source)

    top = np.linalg.norm(source[1] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    width = int(round(max(float(top), float(bottom))))
    height = int(round(max(float(left), float(right))))
    if width < 2 or height < 2:
        raise ValueError("透视矫正目标尺寸必须至少为 2 像素")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    return cv2.warpPerspective(image, transform, (width, height))

def _preprocess_configured_perspective_points(
    config: Mapping[str, Any],
) -> list[tuple[float, float]] | None:
    """Read optional perspective points without changing the base config schema."""
    value: Any = config.get("perspective_points")
    preprocess = config.get("preprocess")
    if value is None and isinstance(preprocess, Mapping):
        value = preprocess.get("perspective_points")
    if value is None:
        return None
    if isinstance(value, str):
        return parse_perspective_points(value)
    return [tuple(map(float, point)) for point in _as_perspective_array(value).tolist()]

def _prepare_roi_mask(
    roi_mask: np.ndarray | None, image_shape: tuple[int, int]
) -> np.ndarray | None:
    if roi_mask is None:
        return None
    if not isinstance(roi_mask, np.ndarray) or roi_mask.ndim != 2:
        raise ValueError("roi_mask must be a two-dimensional array")
    if roi_mask.shape != image_shape:
        raise ValueError("roi_mask dimensions must match the working image")
    _validate_numeric_array(roi_mask, "roi_mask")
    return np.where(roi_mask != 0, 255, 0).astype(np.uint8)

def _opencv_processing_image(image: np.ndarray) -> np.ndarray:
    if image.dtype in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        return image
    converted = image.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("image values are outside the supported numeric range")
    return converted

def _canny_input(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)

def preprocess_image(
    image: np.ndarray,
    config: Mapping[str, Any],
    roi_mask: np.ndarray | None = None,
) -> PreprocessResult:
    """Run rectification, grayscale, blur, optional close, ROI, and Canny."""
    validate_config(config)
    _validate_image(image)

    perspective_points = _preprocess_configured_perspective_points(config)
    working = (
        rectify_perspective(image, perspective_points)
        if perspective_points is not None
        else image.copy()
    )
    processing_image = _opencv_processing_image(working)
    if processing_image.ndim == 2 or (
        processing_image.ndim == 3 and processing_image.shape[2] == 1
    ):
        gray = (
            processing_image
            if processing_image.ndim == 2
            else processing_image[:, :, 0]
        )
    elif processing_image.shape[2] == 3:
        gray = cv2.cvtColor(processing_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(processing_image, cv2.COLOR_BGRA2GRAY)

    kernel_size = int(config["preprocess"]["gaussian_kernel"])
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edge_input = blurred
    preprocess_config = config["preprocess"]
    if preprocess_config["morphology_enabled"]:
        morphology_size = int(preprocess_config["morphology_kernel"])
        morphology_kernel = np.ones(
            (morphology_size, morphology_size), dtype=np.uint8
        )
        edge_input = cv2.morphologyEx(
            edge_input, cv2.MORPH_CLOSE, morphology_kernel
        )

    roi = _prepare_roi_mask(roi_mask, working.shape[:2])
    edges = cv2.Canny(
        _canny_input(edge_input),
        float(preprocess_config["canny_low"]),
        float(preprocess_config["canny_high"]),
    )
    if roi is not None:
        edges = cv2.bitwise_and(edges, edges, mask=roi)

    return PreprocessResult(
        working=working,
        gray=gray,
        blurred=blurred,
        edges=edges,
        rectified=perspective_points is not None,
    )

@dataclass
class SegmentDirectionGroup:
    """Hough segments sharing one undirected orientation."""

    angle_degrees: float
    segments: list[LineSegment]
    total_length: float

@dataclass
class _CenterlineCandidate:
    normal_position: float
    axis_position: float
    support_segments: int
    support_length: float

__all__ = [
    "SegmentDirectionGroup",
    "detect_hough_segments",
    "detect_rebar_lines",
    "group_segments_by_direction",
    "merge_direction_group",
]

def _require_finite_segment(segment: LineSegment) -> None:
    values = (
        segment.x1,
        segment.y1,
        segment.x2,
        segment.y2,
        segment.length,
        segment.angle_degrees,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("segments must contain only finite values")
    if segment.length <= 0:
        raise ValueError("segments must have positive length")

def _normalize_undirected_angle(angle: float) -> float:
    """Normalize angles while collapsing floating-point 180° round-off to 0°."""
    normalized = normalize_angle_degrees(angle)
    return 0.0 if math.isclose(normalized, 180.0, abs_tol=1e-10) else normalized

def _segment_sort_key(segment: LineSegment) -> tuple[float, float, float, float, float, float]:
    """Order equal-length inputs deterministically before incremental grouping."""
    return (
        -float(segment.length),
        _normalize_undirected_angle(float(segment.angle_degrees)),
        float(segment.x1),
        float(segment.y1),
        float(segment.x2),
        float(segment.y2),
    )

def _weighted_group_angle(segments: Sequence[LineSegment]) -> float:
    x = sum(
        segment.length * math.cos(math.radians(2.0 * segment.angle_degrees))
        for segment in segments
    )
    y = sum(
        segment.length * math.sin(math.radians(2.0 * segment.angle_degrees))
        for segment in segments
    )
    if math.isclose(x, 0.0, abs_tol=1e-12) and math.isclose(y, 0.0, abs_tol=1e-12):
        return _normalize_undirected_angle(segments[0].angle_degrees)
    return _normalize_undirected_angle(math.degrees(0.5 * math.atan2(y, x)))

def _update_group(group: SegmentDirectionGroup) -> None:
    group.total_length = float(sum(segment.length for segment in group.segments))
    group.angle_degrees = _weighted_group_angle(group.segments)

def _group_sort_key(group: SegmentDirectionGroup) -> tuple[float, float, int, tuple[tuple[float, ...], ...]]:
    return (
        -group.total_length,
        _normalize_undirected_angle(group.angle_degrees),
        -len(group.segments),
        tuple(
            (segment.x1, segment.y1, segment.x2, segment.y2, segment.length)
            for segment in sorted(group.segments, key=_segment_sort_key)
        ),
    )

def _validate_edge_image(edges: np.ndarray) -> np.ndarray:
    if not isinstance(edges, np.ndarray) or edges.ndim != 2 or edges.size == 0:
        raise ValueError("edges must be a non-empty two-dimensional array")
    if not np.issubdtype(edges.dtype, np.number) or not np.isfinite(edges).all():
        raise ValueError("edges must contain only finite numeric values")
    return np.clip(edges, 0, 255).astype(np.uint8, copy=False)

def detect_hough_segments(edges: np.ndarray, config: Mapping[str, Any]) -> list[LineSegment]:
    """Detect finite line segments from a Canny edge image using HoughLinesP."""
    validate_config(config)
    edge_image = _validate_edge_image(edges)
    hough = config["hough"]
    lines = cv2.HoughLinesP(
        edge_image,
        rho=float(hough["rho"]),
        theta=math.radians(float(hough["theta_degrees"])),
        threshold=int(hough["threshold"]),
        minLineLength=float(hough["min_line_length"]),
        maxLineGap=float(hough["max_line_gap"]),
    )
    if lines is None:
        return []

    segments: list[LineSegment] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        segment = segment_from_points((float(x1), float(y1), float(x2), float(y2)))
        if segment.length > 0 and math.isfinite(segment.length):
            segments.append(segment)
    return sorted(segments, key=_segment_sort_key)

def group_segments_by_direction(
    segments: Sequence[LineSegment], config: Mapping[str, Any]
) -> list[SegmentDirectionGroup]:
    """Group segments by undirected angle using length-weighted circular means."""
    validate_config(config)
    for segment in segments:
        _require_finite_segment(segment)

    tolerance = float(config["grouping"]["angle_tolerance_degrees"])
    groups: list[SegmentDirectionGroup] = []
    for segment in sorted(segments, key=_segment_sort_key):
        matching_groups = [
            (undirected_angle_distance(segment.angle_degrees, group.angle_degrees), index)
            for index, group in enumerate(groups)
        ]
        if matching_groups:
            minimum_distance, group_index = min(matching_groups, key=lambda item: (item[0], item[1]))
        else:
            minimum_distance, group_index = math.inf, -1

        if minimum_distance <= tolerance:
            groups[group_index].segments.append(segment)
            _update_group(groups[group_index])
        else:
            groups.append(
                SegmentDirectionGroup(
                    angle_degrees=_normalize_undirected_angle(segment.angle_degrees),
                    segments=[segment],
                    total_length=float(segment.length),
                )
            )

    grouping = config["grouping"]
    filtered_groups = [
        group
        for group in groups
        if len(group.segments) >= int(grouping["min_group_segments"])
        and group.total_length >= float(grouping["min_group_total_length"])
    ]
    return sorted(filtered_groups, key=_group_sort_key)[: int(grouping["max_direction_groups"])]

def _weighted_average(values: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        raise ValueError("candidate support length must be positive")
    return float(sum(value * weight for value, weight in values) / total_weight)

def _candidate_from_segments(
    segments: Sequence[LineSegment], direction: tuple[float, float], normal: tuple[float, float]
) -> _CenterlineCandidate:
    support_length = float(sum(segment.length for segment in segments))
    normal_positions: list[tuple[float, float]] = []
    axis_positions: list[tuple[float, float]] = []
    for segment in segments:
        midpoint = ((segment.x1 + segment.x2) / 2.0, (segment.y1 + segment.y2) / 2.0)
        normal_positions.append((project_point(midpoint, normal), segment.length))
        axis_positions.append((project_point(midpoint, direction), segment.length))
    return _CenterlineCandidate(
        normal_position=_weighted_average(normal_positions),
        axis_position=_weighted_average(axis_positions),
        support_segments=len(segments),
        support_length=support_length,
    )

def _axis_coverage(segments: Sequence[LineSegment], direction: tuple[float, float]) -> float:
    """Return the union length of segment intervals projected onto the line axis."""
    intervals = sorted(
        (
            min(start, end),
            max(start, end),
        )
        for segment in segments
        for start, end in [
            (
                project_point((segment.x1, segment.y1), direction),
                project_point((segment.x2, segment.y2), direction),
            )
        ]
    )
    coverage = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            coverage += current_end - current_start
            current_start, current_end = start, end
    return float(coverage + current_end - current_start)

def _candidate_sort_key(candidate: _CenterlineCandidate) -> tuple[float, float, int, float]:
    return (
        candidate.normal_position,
        candidate.axis_position,
        candidate.support_segments,
        candidate.support_length,
    )

def _merge_candidate_cluster(candidates: Sequence[_CenterlineCandidate]) -> _CenterlineCandidate:
    return _CenterlineCandidate(
        normal_position=_weighted_average(
            [(candidate.normal_position, candidate.support_length) for candidate in candidates]
        ),
        axis_position=_weighted_average(
            [(candidate.axis_position, candidate.support_length) for candidate in candidates]
        ),
        support_segments=sum(candidate.support_segments for candidate in candidates),
        support_length=float(sum(candidate.support_length for candidate in candidates)),
    )

def _image_dimensions(image_shape: Sequence[int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = image_shape[:2]
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("image dimensions must be positive")
    if not (math.isfinite(float(height)) and math.isfinite(float(width))):
        raise ValueError("image dimensions must be finite")
    if int(height) != height or int(width) != width or height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive integers")
    return int(height), int(width)

def merge_direction_group(
    group: SegmentDirectionGroup,
    image_shape: Sequence[int],
    config: Mapping[str, Any],
    direction_id: str,
) -> list[RebarLine]:
    """Merge parallel Hough-edge segments into clipped, stable rebar centerlines."""
    validate_config(config)
    height, width = _image_dimensions(image_shape)
    if not isinstance(direction_id, str) or not direction_id:
        raise ValueError("direction_id must be a non-empty string")
    if not math.isfinite(float(group.angle_degrees)) or not math.isfinite(float(group.total_length)):
        raise ValueError("group must contain only finite values")
    for segment in group.segments:
        _require_finite_segment(segment)
    if not group.segments:
        return []

    angle = _normalize_undirected_angle(group.angle_degrees)
    direction = direction_vector(angle)
    normal = normal_vector(angle)
    projected_segments = [
        (
            project_point(
                ((segment.x1 + segment.x2) / 2.0, (segment.y1 + segment.y2) / 2.0),
                normal,
            ),
            segment,
        )
        for segment in group.segments
    ]
    projected_segments.sort(key=lambda item: (item[0], _segment_sort_key(item[1])))
    merging = config["merging"]
    cluster_distance = float(merging["normal_cluster_distance_px"])
    clusters: list[list[LineSegment]] = []
    previous_projection: float | None = None
    for projection, segment in projected_segments:
        if previous_projection is None or projection - previous_projection > cluster_distance + 1e-9:
            clusters.append([segment])
        else:
            clusters[-1].append(segment)
        previous_projection = projection

    candidates = [
        _candidate_from_segments(cluster, direction, normal)
        for cluster in clusters
        if len(cluster) >= int(merging["min_support_segments"])
        and sum(segment.length for segment in cluster) >= float(merging["min_support_length"])
        and _axis_coverage(cluster, direction) >= float(merging["min_axis_coverage_px"])
    ]
    if not candidates:
        return []

    duplicate_distance = float(merging["duplicate_center_distance_px"])
    merged_candidates: list[_CenterlineCandidate] = []
    candidate_cluster: list[_CenterlineCandidate] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if not candidate_cluster:
            candidate_cluster = [candidate]
            continue
        previous_candidate = candidate_cluster[-1]
        if (
            candidate.normal_position - previous_candidate.normal_position
            <= duplicate_distance + 1e-9
        ):
            candidate_cluster.append(candidate)
        else:
            merged_candidates.append(_merge_candidate_cluster(candidate_cluster))
            candidate_cluster = [candidate]
    if candidate_cluster:
        merged_candidates.append(_merge_candidate_cluster(candidate_cluster))

    rebars: list[RebarLine] = []
    for candidate in sorted(merged_candidates, key=_candidate_sort_key):
        center = (
            normal[0] * candidate.normal_position + direction[0] * candidate.axis_position,
            normal[1] * candidate.normal_position + direction[1] * candidate.axis_position,
        )
        try:
            p1, p2 = clip_infinite_line_to_image(center, direction, width, height)
        except ValueError:
            continue
        rebars.append(
            RebarLine(
                id=f"{direction_id}-{len(rebars) + 1:02d}",
                direction_id=direction_id,
                p1=p1,
                p2=p2,
                center=center,
                angle_degrees=angle,
                normal_position=candidate.normal_position,
                support_segments=candidate.support_segments,
                support_length=candidate.support_length,
            )
        )
    return rebars

def detect_rebar_lines(
    edges: np.ndarray, config: Mapping[str, Any]
) -> tuple[list[SegmentDirectionGroup], list[list[RebarLine]]]:
    """Detect Hough segments, group their directions, and merge each group."""
    segments = detect_hough_segments(edges, config)
    groups = group_segments_by_direction(segments, config)
    merged_lines = [
        merge_direction_group(group, edges.shape, config, chr(ord("A") + index))
        for index, group in enumerate(groups)
    ]
    return groups, merged_lines

_DIRECTION_BOUNDARY_DEGREES = 22.5

_NO_CALIBRATION_WARNING = "未提供毫米标定，间距单位为像素。"

_LOW_SAMPLE_WARNING = "间距样本较少，统计结果仅供参考。"

_ZERO_MEDIAN_WARNING = "间距中位数为 0，无法执行异常判定。"

__all__ = ["classify_direction_name", "measure_direction"]

def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _validate_measurement_mm_per_pixel(mm_per_pixel: Any) -> float | None:
    if mm_per_pixel is None:
        return None
    value = _finite_float(mm_per_pixel, "mm_per_pixel")
    if value <= 0:
        raise ValueError("mm_per_pixel must be positive")
    return value

def classify_direction_name(angle_degrees: float) -> str:
    """Classify an undirected line angle into a Chinese direction name."""
    angle = normalize_angle_degrees(_finite_float(angle_degrees, "angle_degrees"))
    distance_from_horizontal = min(angle, 180.0 - angle)
    if distance_from_horizontal <= _DIRECTION_BOUNDARY_DEGREES:
        return "横向"
    if abs(angle - 90.0) <= _DIRECTION_BOUNDARY_DEGREES:
        return "纵向"
    return "斜向"

def _rebar_sort_key(rebar: RebarLine) -> tuple[float, str, float, float, float, float, float]:
    """Return a deterministic key for normal-ordering centerlines."""
    return (
        _finite_float(rebar.normal_position, "rebar normal_position"),
        str(rebar.id),
        _finite_float(rebar.center[0], "rebar center"),
        _finite_float(rebar.center[1], "rebar center"),
        _finite_float(rebar.p1[0], "rebar endpoint"),
        _finite_float(rebar.p1[1], "rebar endpoint"),
        _finite_float(rebar.p2[0], "rebar endpoint"),
    )

def _validate_rebars(rebars: Sequence[RebarLine]) -> list[RebarLine]:
    if isinstance(rebars, (str, bytes)):
        raise ValueError("rebars must be a sequence of RebarLine values")
    try:
        ordered = list(rebars)
    except TypeError as exc:
        raise ValueError("rebars must be a sequence of RebarLine values") from exc
    if not all(isinstance(rebar, RebarLine) for rebar in ordered):
        raise ValueError("rebars must contain only RebarLine values")
    return sorted(ordered, key=_rebar_sort_key)

def measure_direction(
    direction_id: str,
    angle_degrees: float,
    rebars: Sequence[RebarLine],
    config: Mapping[str, Any],
    mm_per_pixel: float | None = None,
) -> DirectionResult:
    """Measure adjacent centerline spacing for one detected direction group."""
    validate_config(config)
    if not isinstance(direction_id, str) or not direction_id:
        raise ValueError("direction_id must be a non-empty string")
    angle = _finite_float(angle_degrees, "angle_degrees")
    calibration = _validate_measurement_mm_per_pixel(mm_per_pixel)
    ordered_rebars = _validate_rebars(rebars)

    pixel_spacings = [
        float(next_rebar.normal_position - current_rebar.normal_position)
        for current_rebar, next_rebar in zip(ordered_rebars, ordered_rebars[1:])
    ]
    spacing_median_pixels = median(pixel_spacings) if pixel_spacings else None
    anomaly_ratio = float(config["measurement"]["anomaly_ratio"])
    unit = "mm" if calibration is not None else "px"
    spacings: list[SpacingMeasurement] = []
    for current_rebar, next_rebar, pixels in zip(
        ordered_rebars, ordered_rebars[1:], pixel_spacings
    ):
        value = pixels if calibration is None else pixels * calibration
        if spacing_median_pixels is None or spacing_median_pixels == 0:
            relative_deviation = 0.0
            is_anomaly = False
        else:
            relative_deviation = abs(pixels - spacing_median_pixels) / spacing_median_pixels
            is_anomaly = relative_deviation > anomaly_ratio
        spacings.append(
            SpacingMeasurement(
                from_rebar_id=current_rebar.id,
                to_rebar_id=next_rebar.id,
                pixels=pixels,
                value=float(value),
                unit=unit,
                relative_deviation=float(relative_deviation),
                is_anomaly=is_anomaly,
            )
        )

    values = [spacing.value for spacing in spacings]
    statistics: dict[str, float | int | str | None] = {
        "count": len(values),
        "mean": float(mean(values)) if values else None,
        "median": float(median(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "unit": unit,
        "anomaly_count": sum(spacing.is_anomaly for spacing in spacings),
    }
    warnings: list[str] = []
    if calibration is None:
        warnings.append(_NO_CALIBRATION_WARNING)
    if spacing_median_pixels == 0:
        warnings.append(_ZERO_MEDIAN_WARNING)
    if len(values) < int(config["measurement"]["low_sample_spacing_count"]):
        warnings.append(_LOW_SAMPLE_WARNING)

    return DirectionResult(
        id=direction_id,
        name=classify_direction_name(angle),
        angle_degrees=angle,
        rebars=ordered_rebars,
        spacings=spacings,
        statistics=statistics,
        warnings=warnings,
    )

_GROUP_COLORS = [(255, 120, 0), (0, 200, 255)]

_NORMAL_SPACING_COLOR = (0, 200, 0)

_ANOMALY_SPACING_COLOR = (0, 0, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _as_canvas(image: np.ndarray) -> np.ndarray:
    """Return a BGR drawing copy of a non-empty input image."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty array")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("image must have two or three dimensions")
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("image must have 1, 3, or 4 channels")

def _point(value: Sequence[float]) -> tuple[int, int]:
    """Round a finite image point for OpenCV drawing."""
    x, y = (float(coordinate) for coordinate in value)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("drawing coordinates must be finite")
    return int(round(x)), int(round(y))

def _line_midpoint(rebar: RebarLine) -> tuple[int, int]:
    """Use the midpoint of clipped endpoints for a stable visible arrow anchor."""
    return _point(
        (
            (float(rebar.p1[0]) + float(rebar.p2[0])) / 2.0,
            (float(rebar.p1[1]) + float(rebar.p2[1])) / 2.0,
        )
    )

def _group_color(index: int) -> tuple[int, int, int]:
    return _GROUP_COLORS[index % len(_GROUP_COLORS)]

def draw_raw_segments(
    image: np.ndarray, groups: Sequence[SegmentDirectionGroup]
) -> np.ndarray:
    """Draw retained raw Hough segments in their significant direction colors."""
    canvas = _as_canvas(image)
    for group_index, group in enumerate(groups):
        color = _group_color(group_index)
        for segment in group.segments:
            cv2.line(
                canvas,
                _point((segment.x1, segment.y1)),
                _point((segment.x2, segment.y2)),
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas

def _draw_summary(canvas: np.ndarray, directions: Sequence[DirectionResult]) -> None:
    """Draw ASCII-only summary labels for OpenCV's built-in font."""
    y = 22
    for index, direction in enumerate(directions):
        statistics = direction.statistics
        unit = str(statistics.get("unit", "px"))
        count = len(direction.rebars)
        cv2.putText(
            canvas,
            f"{direction.id} {direction.name}: {count} bars",
            (8, y),
            _FONT,
            0.5,
            _group_color(index),
            1,
            cv2.LINE_AA,
        )
        y += 20
        average = statistics.get("mean")
        if isinstance(average, (int, float)) and math.isfinite(float(average)):
            cv2.putText(
                canvas,
                f"mean: {float(average):.1f} {unit}",
                (8, y),
                _FONT,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18
        anomalies = statistics.get("anomaly_count", 0)
        cv2.putText(
            canvas,
            f"anomalies: {int(anomalies)}",
            (8, y),
            _FONT,
            0.45,
            _ANOMALY_SPACING_COLOR if anomalies else _NORMAL_SPACING_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 22

def draw_detection(image: np.ndarray, directions: Sequence[DirectionResult]) -> np.ndarray:
    """Draw centerlines, IDs, and arrows for exactly the measured adjacent pairs."""
    canvas = _as_canvas(image)
    for group_index, direction in enumerate(directions):
        color = _group_color(group_index)
        rebars_by_id = {rebar.id: rebar for rebar in direction.rebars}
        for rebar in direction.rebars:
            cv2.line(canvas, _point(rebar.p1), _point(rebar.p2), color, 2, cv2.LINE_AA)
            center = _line_midpoint(rebar)
            cv2.putText(
                canvas,
                rebar.id,
                (center[0] + 4, center[1] - 4),
                _FONT,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        for spacing in direction.spacings:
            first = rebars_by_id.get(spacing.from_rebar_id)
            second = rebars_by_id.get(spacing.to_rebar_id)
            if first is None or second is None:
                continue
            start, end = _line_midpoint(first), _line_midpoint(second)
            spacing_color = (
                _ANOMALY_SPACING_COLOR if spacing.is_anomaly else _NORMAL_SPACING_COLOR
            )
            thickness = 2 if spacing.is_anomaly else 1
            cv2.arrowedLine(
                canvas, start, end, spacing_color, thickness, cv2.LINE_AA, tipLength=0.12
            )
            cv2.arrowedLine(
                canvas, end, start, spacing_color, thickness, cv2.LINE_AA, tipLength=0.12
            )
            label_position = ((start[0] + end[0]) // 2 + 3, (start[1] + end[1]) // 2 - 3)
            cv2.putText(
                canvas,
                f"{spacing.value:.1f} {spacing.unit}",
                label_position,
                _FONT,
                0.4,
                spacing_color,
                thickness,
                cv2.LINE_AA,
            )

    _draw_summary(canvas, directions)
    return canvas

def _format_statistic(value: Any, unit: str) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.1f} {unit}"
    return "无"

def format_text_report(report: DetectionReport) -> str:
    """Format a UTF-8 Chinese terminal/report-file summary."""
    lines = [
        f"输入图像：{report.input_path}",
        f"图像尺寸：{report.image_width} × {report.image_height} 像素",
        f"透视矫正：{'是' if report.rectified else '否'}",
        (
            f"毫米标定：{report.mm_per_pixel:.6g} 毫米/像素"
            if report.mm_per_pixel is not None
            else "毫米标定：未提供，间距单位为像素"
        ),
    ]
    if not report.directions:
        lines.append("未检测到可靠的钢筋方向。")

    for direction in report.directions:
        statistics = direction.statistics
        unit = str(statistics.get("unit", "px"))
        lines.extend(
            [
                "",
                f"方向组 {direction.id}：{direction.name}（{direction.angle_degrees:.1f}°）",
                f"检测钢筋：{len(direction.rebars)} 根",
                f"平均钢筋间距：{_format_statistic(statistics.get('mean'), unit)}",
                f"最大间距：{_format_statistic(statistics.get('max'), unit)}",
                f"最小间距：{_format_statistic(statistics.get('min'), unit)}",
                f"中位数间距：{_format_statistic(statistics.get('median'), unit)}",
                f"明显间距异常：{int(statistics.get('anomaly_count', 0))} 处",
            ]
        )
        lines.extend(f"警告：{warning}" for warning in direction.warnings)

    lines.extend(f"警告：{warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"

_NO_DIRECTION_WARNING = "未检测到可靠的钢筋方向，请检查图像质量或调整检测参数。"

def _validate_mm_per_pixel(mm_per_pixel: float | None) -> float | None:
    if mm_per_pixel is None:
        return None
    if isinstance(mm_per_pixel, bool) or not isinstance(mm_per_pixel, (int, float)):
        raise ValueError("mm_per_pixel 必须是正数")
    value = float(mm_per_pixel)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mm_per_pixel 必须是正数")
    return value

def _read_image(path: Path) -> np.ndarray:
    """Decode an image from bytes so Windows Unicode paths are supported."""
    if not path.is_file():
        raise FileNotFoundError(f"输入图像不存在：{path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"无法读取输入图像：{path}")
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"无法读取输入图像：{path}") from exc
    if image is None or image.size == 0:
        raise ValueError(f"无法读取输入图像：{path}")
    return image

def _write_image(path: Path, image: np.ndarray) -> None:
    """Encode to bytes before writing, avoiding OpenCV Unicode-path limitations."""
    suffix = path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"无法编码输出图像：{path}")
    path.write_bytes(encoded.tobytes())

def _effective_config(
    config: Mapping[str, Any], perspective_points: Sequence[Sequence[float]] | None
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    result = copy.deepcopy(dict(config))
    if perspective_points is not None:
        result["perspective_points"] = perspective_points
    validate_config(result)
    return result

def _configured_perspective_points(config: Mapping[str, Any]) -> Any:
    points = config.get("perspective_points")
    preprocess = config.get("preprocess")
    if points is None and isinstance(preprocess, Mapping):
        points = preprocess.get("perspective_points")
    return parse_perspective_points(points) if isinstance(points, str) else points

def _validate_perspective_bounds(config: Mapping[str, Any], image: np.ndarray) -> None:
    points = _configured_perspective_points(config)
    if points is None:
        return
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("透视点坐标必须是数字") from exc
    if coordinates.shape != (4, 2) or not np.isfinite(coordinates).all():
        raise ValueError("透视矫正必须提供四个有限坐标点")
    height, width = image.shape[:2]
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    if (
        np.any(x_coordinates < 0)
        or np.any(x_coordinates >= width)
        or np.any(y_coordinates < 0)
        or np.any(y_coordinates >= height)
    ):
        raise ValueError("透视点必须位于输入图像范围内")

def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    mm_per_pixel: float | None = None,
    perspective_points: Sequence[Sequence[float]] | None = None,
) -> DetectionReport:
    """Run detection, save teaching images and reports, and return the result."""
    calibration = _validate_mm_per_pixel(mm_per_pixel)
    active_config = _effective_config(config, perspective_points)
    source_path = Path(input_path)
    image = _read_image(source_path)
    _validate_perspective_bounds(active_config, image)

    # Preprocess before output creation: invalid perspective geometry cannot produce
    # a partial, potentially misleading report directory.
    preprocessed = preprocess_image(image, active_config)
    destination = Path(output_dir)
    debug_dir = destination / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if preprocessed.rectified:
        _write_image(debug_dir / "00_rectified.png", preprocessed.working)
    _write_image(debug_dir / "01_gray.png", preprocessed.gray)
    _write_image(debug_dir / "02_blur.png", preprocessed.blurred)
    _write_image(debug_dir / "03_edges.png", preprocessed.edges)

    raw_segments = detect_hough_segments(preprocessed.edges, active_config)
    groups = group_segments_by_direction(raw_segments, active_config)
    _write_image(debug_dir / "04_hough_raw.png", draw_raw_segments(preprocessed.working, groups))

    directions = []
    for index, group in enumerate(groups):
        direction_id = chr(ord("A") + index)
        rebars = merge_direction_group(
            group, preprocessed.edges.shape, active_config, direction_id
        )
        if rebars:
            directions.append(
                measure_direction(
                    direction_id,
                    group.angle_degrees,
                    rebars,
                    active_config,
                    calibration,
                )
            )

    _write_image(debug_dir / "05_merged_lines.png", draw_detection(preprocessed.working, directions))
    warnings = [] if directions else [_NO_DIRECTION_WARNING]
    report = DetectionReport(
        input_path=str(source_path),
        output_path=str(destination),
        image_width=int(preprocessed.working.shape[1]),
        image_height=int(preprocessed.working.shape[0]),
        rectified=preprocessed.rectified,
        mm_per_pixel=calibration,
        parameters=active_config,
        directions=directions,
        warnings=warnings,
    )
    _write_image(destination / "annotated.png", draw_detection(preprocessed.working, directions))
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (destination / "report.txt").write_text(format_text_report(report), encoding="utf-8")
    return report

class _FriendlyArgumentParser(argparse.ArgumentParser):
    """Raise validation errors so the CLI emits a consistent Chinese message."""

    def error(self, message: str) -> None:
        raise ValueError(message)


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mm-per-pixel 必须是正数") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("mm-per-pixel 必须是正数")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = _FriendlyArgumentParser(description="钢筋检测与间距分析")
    parser.add_argument(
        "--input",
        default=str(INPUT_IMAGE_PATH),
        help=f"输入图像路径，默认读取 {INPUT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"输出目录，默认写入 {OUTPUT_DIR}",
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置路径")
    parser.add_argument(
        "--mm-per-pixel",
        type=_positive_float,
        default=MM_PER_PIXEL,
        help="标定比例（毫米/像素，必须大于 0）",
    )
    parser.add_argument(
        "--perspective-points",
        default=PERSPECTIVE_POINTS,
        help="透视点：左上;右上;右下;左下，格式 x,y;x,y;x,y;x,y",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the detector CLI and return a conventional process exit code."""
    try:
        arguments = _build_parser().parse_args(argv)
        config = load_config(arguments.config)
        points = parse_perspective_points(arguments.perspective_points)
        report = run_pipeline(
            arguments.input,
            arguments.output,
            config,
            mm_per_pixel=arguments.mm_per_pixel,
            perspective_points=points,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(format_text_report(report), end="")
    print(f"输出目录：{arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

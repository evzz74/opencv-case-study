"""Perspective correction and teachable image-preprocessing stages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import validate_config


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


def _configured_perspective_points(
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

    perspective_points = _configured_perspective_points(config)
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

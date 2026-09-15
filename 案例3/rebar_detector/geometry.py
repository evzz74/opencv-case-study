"""Pure geometric helpers used by the rebar detection pipeline."""

from __future__ import annotations

import math
from typing import Sequence

from .models import LineSegment


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

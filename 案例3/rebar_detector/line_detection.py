"""Hough-segment detection, direction grouping, and rebar centerline merging."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import validate_config
from .geometry import (
    clip_infinite_line_to_image,
    direction_vector,
    normal_vector,
    normalize_angle_degrees,
    project_point,
    segment_from_points,
    undirected_angle_distance,
)
from .models import LineSegment, RebarLine


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

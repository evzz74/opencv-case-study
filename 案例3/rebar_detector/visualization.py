"""Drawing and text-report helpers for rebar detection results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .line_detection import SegmentDirectionGroup
from .models import DetectionReport, DirectionResult, RebarLine

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

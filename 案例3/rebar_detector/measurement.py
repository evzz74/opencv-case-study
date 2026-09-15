"""Spacing measurement, unit conversion, and anomaly analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from .config import validate_config
from .geometry import normalize_angle_degrees
from .models import DirectionResult, RebarLine, SpacingMeasurement


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


def _validate_mm_per_pixel(mm_per_pixel: Any) -> float | None:
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
    calibration = _validate_mm_per_pixel(mm_per_pixel)
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

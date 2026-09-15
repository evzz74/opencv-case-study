"""Loading and validation for the detector configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

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

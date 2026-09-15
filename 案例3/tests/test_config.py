import json
from pathlib import Path

import pytest

from rebar_detector.config import load_config, validate_config
from rebar_detector.models import (
    DetectionReport,
    DirectionResult,
    LineSegment,
    RebarLine,
    SpacingMeasurement,
)


DEFAULT_CONFIG = Path("config/default.yaml")


def test_load_default_config_contains_expected_values():
    config = load_config(DEFAULT_CONFIG)
    assert config["preprocess"]["gaussian_kernel"] == 5
    assert config["measurement"]["anomaly_ratio"] == pytest.approx(0.20)


def test_load_config_without_path_uses_project_default():
    assert load_config()["hough"]["threshold"] == 110


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("preprocess", "gaussian_kernel", 4),
        ("preprocess", "canny_low", 200),
        ("measurement", "anomaly_ratio", 0),
        ("grouping", "max_direction_groups", 0),
    ],
)
def test_validate_config_rejects_invalid_values(section, key, value):
    config = load_config(DEFAULT_CONFIG)
    config[section][key] = value
    if key == "canny_low":
        config["preprocess"]["canny_high"] = 100
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("section", ["preprocess", "hough", "merging", "output"])
def test_validate_config_rejects_missing_required_section(section):
    config = load_config(DEFAULT_CONFIG)
    del config[section]
    with pytest.raises(ValueError, match="missing mapping"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("preprocess", "canny_high"),
        ("hough", "threshold"),
        ("grouping", "min_group_segments"),
        ("measurement", "anomaly_ratio"),
    ],
)
def test_validate_config_rejects_missing_required_key(section, key):
    config = load_config(DEFAULT_CONFIG)
    del config[section][key]
    with pytest.raises(ValueError, match="missing .* keys"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("preprocess", "gaussian_kernel", True),
        ("preprocess", "morphology_enabled", 1),
        ("grouping", "max_direction_groups", 1.5),
        ("merging", "min_support_segments", 1.5),
        ("output", "save_debug", "true"),
    ],
)
def test_validate_config_rejects_invalid_types(section, key, value):
    config = load_config(DEFAULT_CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("kernel", [0, 2, 4, -1])
def test_validate_config_rejects_non_positive_or_even_morphology_kernel(kernel):
    config = load_config(DEFAULT_CONFIG)
    config["preprocess"]["morphology_kernel"] = kernel
    with pytest.raises(ValueError, match="morphology_kernel"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("hough", "rho", 0),
        ("hough", "min_line_length", -1),
        ("merging", "normal_cluster_distance_px", 0),
        ("merging", "duplicate_center_distance_px", -0.01),
        ("merging", "min_support_segments", 0),
        ("merging", "min_axis_coverage_px", -1),
    ],
)
def test_validate_config_rejects_non_positive_hough_and_merging_values(
    section, key, value
):
    config = load_config(DEFAULT_CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError, match="positive"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("hough", "rho", float("nan")),
        ("hough", "theta_degrees", float("inf")),
        ("grouping", "angle_tolerance_degrees", float("-inf")),
        ("merging", "normal_cluster_distance_px", float("nan")),
        ("merging", "duplicate_center_distance_px", float("inf")),
    ],
)
def test_validate_config_rejects_non_finite_positive_numeric_values(
    section, key, value
):
    config = load_config(DEFAULT_CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("preprocess", "gaussian_kernel", 1),
        ("preprocess", "morphology_kernel", 1),
        ("preprocess", "canny_low", 0),
        ("grouping", "max_direction_groups", 1),
        ("grouping", "max_direction_groups", 2),
        ("hough", "rho", 1e-12),
        ("merging", "normal_cluster_distance_px", 1e-12),
        ("measurement", "anomaly_ratio", 1e-12),
        ("measurement", "anomaly_ratio", 1 - 1e-12),
        ("measurement", "low_sample_spacing_count", 0),
    ],
)
def test_validate_config_accepts_meaningful_boundary_values(section, key, value):
    config = load_config(DEFAULT_CONFIG)
    config[section][key] = value
    validate_config(config)


def test_detection_report_to_dict_serializes_nested_dataclasses_and_tuples():
    rebar = RebarLine(
        id="r1",
        direction_id="d1",
        p1=(1.0, 2.0),
        p2=(3.0, 4.0),
        center=(2.0, 3.0),
        angle_degrees=45.0,
        normal_position=2.5,
        support_segments=2,
        support_length=12.0,
    )
    spacing = SpacingMeasurement(
        from_rebar_id="r1",
        to_rebar_id="r2",
        pixels=10.0,
        value=10.0,
        unit="px",
        relative_deviation=0.25,
        is_anomaly=True,
    )
    direction = DirectionResult(
        id="d1",
        name="primary",
        angle_degrees=45.0,
        rebars=[rebar],
        spacings=[spacing],
        statistics={"median": 10.0, "count": 1, "note": None},
        warnings=["low sample count"],
    )
    report = DetectionReport(
        input_path="input.png",
        output_path="output",
        image_width=640,
        image_height=480,
        rectified=False,
        mm_per_pixel=None,
        parameters={"grouping": {"max_direction_groups": 2}},
        directions=[direction],
        warnings=[],
    )

    result = report.to_dict()

    assert result["output_path"] == "output"
    assert result["directions"][0]["rebars"][0]["p1"] == [1.0, 2.0]
    assert result["directions"][0]["spacings"][0]["unit"] == "px"
    assert result["directions"][0]["statistics"]["note"] is None
    assert json.loads(json.dumps(result)) == result


def test_line_segment_to_dict_contains_all_fields():
    segment = LineSegment(1.0, 2.0, 4.0, 6.0, 5.0, 53.13)
    assert segment.to_dict() == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 4.0,
        "y2": 6.0,
        "length": 5.0,
        "angle_degrees": 53.13,
    }

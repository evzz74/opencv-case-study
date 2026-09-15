import pytest

from rebar_detector.config import load_config
from rebar_detector.measurement import measure_direction
from rebar_detector.models import RebarLine


def make_rebar(identifier, normal_position):
    return RebarLine(
        id=identifier,
        direction_id="A",
        p1=(normal_position, 0),
        p2=(normal_position, 199),
        center=(normal_position, 99.5),
        angle_degrees=90.0,
        normal_position=normal_position,
        support_segments=2,
        support_length=398.0,
    )


def test_uniform_pixel_spacing_statistics():
    result = measure_direction("A", 90.0, [make_rebar("r1", 10), make_rebar("r2", 50), make_rebar("r3", 90)], load_config("config/default.yaml"))
    assert [spacing.value for spacing in result.spacings] == pytest.approx([40, 40])
    assert result.statistics["mean"] == pytest.approx(40)
    assert result.statistics["unit"] == "px"


def test_mm_per_pixel_converts_values():
    result = measure_direction("A", 90.0, [make_rebar("r1", 10), make_rebar("r2", 50)], load_config("config/default.yaml"), mm_per_pixel=2.5)
    assert result.spacings[0].pixels == pytest.approx(40)
    assert result.spacings[0].value == pytest.approx(100)
    assert result.spacings[0].unit == "mm"


def test_large_gap_is_flagged_against_median():
    rebars = [make_rebar("r1", 0), make_rebar("r2", 40), make_rebar("r3", 80), make_rebar("r4", 140), make_rebar("r5", 180)]
    result = measure_direction("A", 90.0, rebars, load_config("config/default.yaml"))
    assert [spacing.is_anomaly for spacing in result.spacings] == [False, False, True, False]
    assert result.spacings[2].relative_deviation == pytest.approx(0.5)


def test_single_rebar_has_no_spacing_statistics():
    result = measure_direction("A", 90.0, [make_rebar("r1", 10)], load_config("config/default.yaml"))
    assert result.spacings == []
    assert result.statistics["mean"] is None


def test_all_zero_spacings_skip_anomaly_detection_with_warning():
    rebars = [make_rebar(f"r{index}", 10) for index in range(1, 5)]

    result = measure_direction("A", 90.0, rebars, load_config("config/default.yaml"))

    assert [spacing.pixels for spacing in result.spacings] == pytest.approx([0, 0, 0])
    assert [spacing.relative_deviation for spacing in result.spacings] == pytest.approx([0, 0, 0])
    assert [spacing.is_anomaly for spacing in result.spacings] == [False, False, False]
    assert "间距中位数为 0，无法执行异常判定。" in result.warnings


def test_zero_median_mixed_spacings_skip_all_anomaly_detection_with_warning():
    rebars = [
        make_rebar("r1", 0),
        make_rebar("r2", 0),
        make_rebar("r3", 0),
        make_rebar("r4", 0),
        make_rebar("r5", 10),
    ]

    result = measure_direction("A", 90.0, rebars, load_config("config/default.yaml"))

    assert [spacing.pixels for spacing in result.spacings] == pytest.approx([0, 0, 0, 10])
    assert [spacing.relative_deviation for spacing in result.spacings] == pytest.approx([0, 0, 0, 0])
    assert [spacing.is_anomaly for spacing in result.spacings] == [False, False, False, False]
    assert "间距中位数为 0，无法执行异常判定。" in result.warnings

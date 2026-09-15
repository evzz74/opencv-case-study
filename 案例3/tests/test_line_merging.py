import math

import pytest

from conftest import make_grid_image
from rebar_detector.config import load_config
from rebar_detector.geometry import segment_from_points
from rebar_detector.line_detection import (
    detect_hough_segments,
    group_segments_by_direction,
    merge_direction_group,
)
from rebar_detector.preprocess import preprocess_image


def _unit_config():
    """Keep geometry tests independent from production-image tuning defaults."""
    config = load_config("config/default.yaml")
    config["grouping"].update(min_group_segments=2, min_group_total_length=80.0)
    config["merging"].update(
        normal_cluster_distance_px=8.0,
        duplicate_center_distance_px=12.0,
        min_support_segments=1,
        min_support_length=35.0,
        min_axis_coverage_px=25.0,
    )
    return config


def test_groups_horizontal_and_vertical_segments():
    config = _unit_config()
    segments = [
        segment_from_points((10, 20, 200, 20)),
        segment_from_points((10, 80, 200, 80)),
        segment_from_points((30, 10, 30, 180)),
        segment_from_points((90, 10, 90, 180)),
    ]

    groups = group_segments_by_direction(segments, config)

    assert len(groups) == 2
    assert sorted(len(group.segments) for group in groups) == [2, 2]


def test_groups_segments_across_zero_and_180_degree_wrap():
    config = _unit_config()
    segments = [
        segment_from_points((10, 10, 210, 13.5)),
        segment_from_points((10, 40, 210, 36.5)),
    ]

    groups = group_segments_by_direction(segments, config)

    assert len(groups) == 1
    assert groups[0].angle_degrees == pytest.approx(0, abs=2)


def test_two_edges_of_one_thick_bar_merge_to_one_centerline():
    config = _unit_config()
    config["merging"]["normal_cluster_distance_px"] = 4
    config["merging"]["duplicate_center_distance_px"] = 10
    segments = [
        segment_from_points((47, 10, 47, 190)),
        segment_from_points((53, 10, 53, 190)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert len(rebars) == 1
    assert rebars[0].center[0] == pytest.approx(50, abs=1)


def test_separated_bars_remain_separate():
    config = _unit_config()
    segments = [
        segment_from_points((47, 10, 47, 190)),
        segment_from_points((53, 10, 53, 190)),
        segment_from_points((97, 10, 97, 190)),
        segment_from_points((103, 10, 103, 190)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert len(rebars) == 2


def test_fragmented_line_has_combined_axis_coverage():
    config = _unit_config()
    config["merging"]["normal_cluster_distance_px"] = 4
    config["merging"]["duplicate_center_distance_px"] = 4
    config["merging"]["min_support_length"] = 100
    config["merging"]["min_axis_coverage_px"] = 150
    segments = [
        segment_from_points((50, 10, 50, 90)),
        segment_from_points((50, 110, 50, 190)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert len(rebars) == 1
    assert rebars[0].support_segments == 2
    assert rebars[0].support_length == pytest.approx(160)


def test_widely_separated_short_fragments_do_not_fake_axis_coverage():
    config = _unit_config()
    config["grouping"]["min_group_total_length"] = 30
    config["merging"]["min_support_length"] = 30
    config["merging"]["min_axis_coverage_px"] = 50
    segments = [
        segment_from_points((50, 0, 50, 20)),
        segment_from_points((50, 180, 50, 200)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (220, 200), config, "A")

    assert rebars == []


def test_duplicate_candidates_chain_by_adjacent_positions():
    config = _unit_config()
    config["merging"]["normal_cluster_distance_px"] = 1
    config["merging"]["duplicate_center_distance_px"] = 12
    segments = [
        segment_from_points((10, 0, 190, 0)),
        segment_from_points((10, 12, 190, 12)),
        segment_from_points((10, 24, 190, 24)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert len(rebars) == 1
    assert rebars[0].normal_position == pytest.approx(12)
    assert rebars[0].center[1] == pytest.approx(12)


def test_equal_normal_projections_cluster_deterministically():
    config = _unit_config()
    segments = [
        segment_from_points((100, 50, 180, 50)),
        segment_from_points((0, 50, 80, 50)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (120, 200), config, "A")

    assert len(rebars) == 1
    assert rebars[0].support_segments == 2


def test_centerlines_are_sorted_and_assigned_stable_ids():
    config = _unit_config()
    config["merging"]["normal_cluster_distance_px"] = 4
    config["merging"]["duplicate_center_distance_px"] = 10
    segments = [
        segment_from_points((103, 10, 103, 190)),
        segment_from_points((97, 10, 97, 190)),
        segment_from_points((53, 10, 53, 190)),
        segment_from_points((47, 10, 47, 190)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert [rebar.id for rebar in rebars] == ["A-01", "A-02"]
    assert [rebar.normal_position for rebar in rebars] == sorted(
        rebar.normal_position for rebar in rebars
    )
    assert [rebar.center[0] for rebar in rebars] == pytest.approx([100, 50])


def test_hough_detection_finds_usable_grid_segments_in_two_directions():
    config = _unit_config()
    preprocessed = preprocess_image(make_grid_image(), config)

    segments = detect_hough_segments(preprocessed.edges, config)
    groups = group_segments_by_direction(segments, config)

    assert len(segments) >= 4
    assert len(groups) >= 2
    assert all(math.isfinite(segment.length) for segment in segments)


def test_clipped_out_candidates_do_not_create_id_gaps():
    config = _unit_config()
    segments = [
        segment_from_points((210, 10, 210, 190)),
        segment_from_points((50, 10, 50, 190)),
    ]
    group = group_segments_by_direction(segments, config)[0]

    rebars = merge_direction_group(group, (200, 200), config, "A")

    assert [rebar.id for rebar in rebars] == ["A-01"]

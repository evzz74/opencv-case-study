import pytest

from rebar_detector.geometry import (
    clip_infinite_line_to_image,
    direction_vector,
    normal_vector,
    normalize_angle_degrees,
    project_point,
    segment_from_points,
    undirected_angle_distance,
)


def test_undirected_angles_wrap_at_180_degrees():
    assert normalize_angle_degrees(181) == pytest.approx(1)
    assert undirected_angle_distance(179, 1) == pytest.approx(2)


def test_segment_from_points_reports_length_and_angle():
    segment = segment_from_points((0, 0, 3, 4))
    assert segment.length == pytest.approx(5)
    assert segment.angle_degrees == pytest.approx(53.130102, rel=1e-5)


def test_projection_uses_supplied_unit_axis():
    assert project_point((10, 3), (1, 0)) == pytest.approx(10)


def test_direction_and_normal_vectors_are_perpendicular():
    assert direction_vector(0) == pytest.approx((1, 0))
    assert normal_vector(0) == pytest.approx((0, 1))
    assert direction_vector(90) == pytest.approx((0, 1))
    assert normal_vector(90) == pytest.approx((-1, 0))


def test_clip_vertical_line_to_image():
    p1, p2 = clip_infinite_line_to_image((20, 10), (0, 1), 100, 80)
    assert p1 == pytest.approx((20, 0))
    assert p2 == pytest.approx((20, 79))


def test_clip_rejects_zero_direction():
    with pytest.raises(ValueError, match="direction"):
        clip_infinite_line_to_image((20, 10), (0, 0), 100, 80)


def test_clip_rejects_line_without_two_image_intersections():
    with pytest.raises(ValueError, match="intersections"):
        clip_infinite_line_to_image((200, 200), (1, 0), 100, 80)

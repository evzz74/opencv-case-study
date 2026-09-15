import cv2
import numpy as np
import pytest

from rebar_detector.config import load_config
from rebar_detector.preprocess import parse_perspective_points, preprocess_image, rectify_perspective


def test_parse_four_perspective_points():
    points = parse_perspective_points("0,0;99,0;99,49;0,49")
    assert points == [(0.0, 0.0), (99.0, 0.0), (99.0, 49.0), (0.0, 49.0)]


def test_parse_rejects_wrong_point_count():
    with pytest.raises(ValueError, match="四个点"):
        parse_perspective_points("0,0;10,0;10,10")


def test_rectification_preserves_rectangle_dimensions():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    result = rectify_perspective(image, [(0, 0), (99, 0), (99, 49), (0, 49)])
    assert result.shape[:2] == (49, 99)


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (99, 0), (99, 0), (0, 49)],
        [(0, 0), (99, 0), (99, 49), (0, 0)],
    ],
)
def test_rectification_rejects_duplicate_points(points):
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="重复"):
        rectify_perspective(image, points)


def test_rectification_rejects_collinear_corner_points():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="共线|退化"):
        rectify_perspective(image, [(0, 0), (50, 0), (99, 0), (0, 49)])


def test_rectification_rejects_near_zero_area_quadrilateral():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    points = [(0, 0), (100, 0), (200, 1e-8), (-100, 1e-8)]
    with pytest.raises(ValueError, match="面积|退化"):
        rectify_perspective(image, points)


def test_rectification_rejects_self_intersecting_quadrilateral():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    points = [(0, 0), (99, 49), (99, 0), (0, 49)]
    with pytest.raises(ValueError, match="相交"):
        rectify_perspective(image, points)


def test_preprocess_finds_simple_line_edges():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.line(image, (20, 10), (20, 110), (255, 255, 255), 4)
    result = preprocess_image(image, load_config("config/default.yaml"))
    assert result.gray.shape == (120, 120)
    assert np.count_nonzero(result.edges) > 0


def test_preprocess_accepts_integer_image_arrays():
    image = np.zeros((40, 40), dtype=np.uint8)
    result = preprocess_image(image, load_config("config/default.yaml"))
    assert result.edges.dtype == np.uint8


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_preprocess_rejects_nonfinite_image_arrays(value):
    image = np.zeros((40, 40), dtype=np.float32)
    image[10, 10] = value
    with pytest.raises(ValueError, match="image.*finite"):
        preprocess_image(image, load_config("config/default.yaml"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_preprocess_rejects_nonfinite_roi_masks(value):
    image = np.zeros((40, 40), dtype=np.uint8)
    roi_mask = np.ones((40, 40), dtype=np.float32)
    roi_mask[10, 10] = value
    with pytest.raises(ValueError, match="roi_mask.*finite"):
        preprocess_image(image, load_config("config/default.yaml"), roi_mask)


def test_uniform_image_with_interior_roi_has_no_boundary_edges():
    image = np.full((80, 80, 3), 128, dtype=np.uint8)
    roi_mask = np.zeros((80, 80), dtype=np.uint8)
    roi_mask[20:60, 20:60] = 255
    result = preprocess_image(image, load_config("config/default.yaml"), roi_mask)
    assert np.count_nonzero(result.edges) == 0


def test_roi_masks_edges_outside_selected_region():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(image, (25, 10), (25, 90), (255, 255, 255), 3)
    cv2.line(image, (75, 10), (75, 90), (255, 255, 255), 3)
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[:, :50] = 255
    result = preprocess_image(image, load_config("config/default.yaml"), roi_mask)
    assert np.count_nonzero(result.edges[:, :50]) > 0
    assert np.count_nonzero(result.edges[:, 50:]) == 0


def test_preprocess_runs_morphology_close_when_enabled():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.line(image, (10, 40), (70, 40), (255, 255, 255), 3)
    config = load_config("config/default.yaml")
    config["preprocess"]["morphology_enabled"] = True
    result = preprocess_image(image, config)
    assert result.edges.shape == (80, 80)
    assert np.count_nonzero(result.edges) > 0

"""End-to-end rebar detection orchestration and Unicode-safe file output."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import validate_config
from .line_detection import (
    detect_hough_segments,
    group_segments_by_direction,
    merge_direction_group,
)
from .measurement import measure_direction
from .models import DetectionReport
from .preprocess import parse_perspective_points, preprocess_image
from .visualization import draw_detection, draw_raw_segments, format_text_report

_NO_DIRECTION_WARNING = "未检测到可靠的钢筋方向，请检查图像质量或调整检测参数。"


def _validate_mm_per_pixel(mm_per_pixel: float | None) -> float | None:
    if mm_per_pixel is None:
        return None
    if isinstance(mm_per_pixel, bool) or not isinstance(mm_per_pixel, (int, float)):
        raise ValueError("mm_per_pixel 必须是正数")
    value = float(mm_per_pixel)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mm_per_pixel 必须是正数")
    return value


def _read_image(path: Path) -> np.ndarray:
    """Decode an image from bytes so Windows Unicode paths are supported."""
    if not path.is_file():
        raise FileNotFoundError(f"输入图像不存在：{path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"无法读取输入图像：{path}")
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"无法读取输入图像：{path}") from exc
    if image is None or image.size == 0:
        raise ValueError(f"无法读取输入图像：{path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    """Encode to bytes before writing, avoiding OpenCV Unicode-path limitations."""
    suffix = path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"无法编码输出图像：{path}")
    path.write_bytes(encoded.tobytes())


def _effective_config(
    config: Mapping[str, Any], perspective_points: Sequence[Sequence[float]] | None
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    result = copy.deepcopy(dict(config))
    if perspective_points is not None:
        result["perspective_points"] = perspective_points
    validate_config(result)
    return result


def _configured_perspective_points(config: Mapping[str, Any]) -> Any:
    points = config.get("perspective_points")
    preprocess = config.get("preprocess")
    if points is None and isinstance(preprocess, Mapping):
        points = preprocess.get("perspective_points")
    return parse_perspective_points(points) if isinstance(points, str) else points


def _validate_perspective_bounds(config: Mapping[str, Any], image: np.ndarray) -> None:
    points = _configured_perspective_points(config)
    if points is None:
        return
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("透视点坐标必须是数字") from exc
    if coordinates.shape != (4, 2) or not np.isfinite(coordinates).all():
        raise ValueError("透视矫正必须提供四个有限坐标点")
    height, width = image.shape[:2]
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    if (
        np.any(x_coordinates < 0)
        or np.any(x_coordinates >= width)
        or np.any(y_coordinates < 0)
        or np.any(y_coordinates >= height)
    ):
        raise ValueError("透视点必须位于输入图像范围内")


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    mm_per_pixel: float | None = None,
    perspective_points: Sequence[Sequence[float]] | None = None,
) -> DetectionReport:
    """Run detection, save teaching images and reports, and return the result."""
    calibration = _validate_mm_per_pixel(mm_per_pixel)
    active_config = _effective_config(config, perspective_points)
    source_path = Path(input_path)
    image = _read_image(source_path)
    _validate_perspective_bounds(active_config, image)

    # Preprocess before output creation: invalid perspective geometry cannot produce
    # a partial, potentially misleading report directory.
    preprocessed = preprocess_image(image, active_config)
    destination = Path(output_dir)
    debug_dir = destination / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if preprocessed.rectified:
        _write_image(debug_dir / "00_rectified.png", preprocessed.working)
    _write_image(debug_dir / "01_gray.png", preprocessed.gray)
    _write_image(debug_dir / "02_blur.png", preprocessed.blurred)
    _write_image(debug_dir / "03_edges.png", preprocessed.edges)

    raw_segments = detect_hough_segments(preprocessed.edges, active_config)
    groups = group_segments_by_direction(raw_segments, active_config)
    _write_image(debug_dir / "04_hough_raw.png", draw_raw_segments(preprocessed.working, groups))

    directions = []
    for index, group in enumerate(groups):
        direction_id = chr(ord("A") + index)
        rebars = merge_direction_group(
            group, preprocessed.edges.shape, active_config, direction_id
        )
        if rebars:
            directions.append(
                measure_direction(
                    direction_id,
                    group.angle_degrees,
                    rebars,
                    active_config,
                    calibration,
                )
            )

    _write_image(debug_dir / "05_merged_lines.png", draw_detection(preprocessed.working, directions))
    warnings = [] if directions else [_NO_DIRECTION_WARNING]
    report = DetectionReport(
        input_path=str(source_path),
        output_path=str(destination),
        image_width=int(preprocessed.working.shape[1]),
        image_height=int(preprocessed.working.shape[0]),
        rectified=preprocessed.rectified,
        mm_per_pixel=calibration,
        parameters=active_config,
        directions=directions,
        warnings=warnings,
    )
    _write_image(destination / "annotated.png", draw_detection(preprocessed.working, directions))
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (destination / "report.txt").write_text(format_text_report(report), encoding="utf-8")
    return report

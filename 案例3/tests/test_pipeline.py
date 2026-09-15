import json

import cv2
import numpy as np
import pytest

from main import main
from rebar_detector.config import load_config
from rebar_detector.pipeline import run_pipeline


def test_pipeline_writes_required_outputs(tmp_path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in (50, 100, 150, 200, 250):
        cv2.line(image, (x, 15), (x, 225), (255, 255, 255), 5)
    input_path = tmp_path / "grid.png"
    output_dir = tmp_path / "output"
    cv2.imwrite(str(input_path), image)

    report = run_pipeline(input_path, output_dir, load_config("config/default.yaml"))

    assert (output_dir / "annotated.png").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.txt").exists()
    assert (output_dir / "debug" / "01_gray.png").exists()
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["mm_per_pixel"] is None
    assert payload["output_path"] == str(output_dir)
    assert "mm" not in (output_dir / "report.txt").read_text(encoding="utf-8")
    assert report.image_width == 320
    assert report.output_path == str(output_dir)


def test_blank_image_writes_empty_report_with_warning(tmp_path):
    input_path = tmp_path / "blank.png"
    output_dir = tmp_path / "output"
    cv2.imwrite(str(input_path), np.zeros((120, 120, 3), dtype=np.uint8))
    report = run_pipeline(input_path, output_dir, load_config("config/default.yaml"))
    assert report.directions == []
    assert any("未检测到" in warning for warning in report.warnings)
    assert (output_dir / "debug" / "03_edges.png").exists()


def test_cli_prints_chinese_report_and_rejects_invalid_calibration(tmp_path, capsys):
    input_path = tmp_path / "blank.png"
    output_dir = tmp_path / "output"
    cv2.imwrite(str(input_path), np.zeros((120, 120, 3), dtype=np.uint8))

    assert main(["--input", str(input_path), "--output", str(output_dir)]) == 0
    captured = capsys.readouterr()
    assert "未检测到" in captured.out
    assert str(output_dir) in captured.out

    assert main(["--input", str(input_path), "--mm-per-pixel", "0"]) == 2
    assert "错误：" in capsys.readouterr().err


@pytest.mark.parametrize("contents", [b"", b"not an image"])
def test_unreadable_image_has_friendly_pipeline_and_cli_errors(
    tmp_path, capsys, contents
):
    input_path = tmp_path / "损坏图像.png"
    output_dir = tmp_path / "output"
    input_path.write_bytes(contents)

    with pytest.raises(ValueError, match="无法读取输入图像"):
        run_pipeline(input_path, output_dir, load_config("config/default.yaml"))
    assert not output_dir.exists()

    assert main(["--input", str(input_path), "--output", str(output_dir)]) == 2
    captured = capsys.readouterr()
    assert "错误：无法读取输入图像" in captured.err
    assert "Traceback" not in captured.err
    assert not output_dir.exists()


def test_perspective_points_must_be_inside_source_image(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    input_path = tmp_path / "source.png"
    output_dir = tmp_path / "output"
    cv2.imwrite(str(input_path), image)

    with pytest.raises(ValueError, match="透视点必须位于输入图像范围内"):
        run_pipeline(
            input_path,
            output_dir,
            load_config("config/default.yaml"),
            perspective_points=[(0, 0), (160, 0), (159, 119), (0, 119)],
        )

    assert not output_dir.exists()

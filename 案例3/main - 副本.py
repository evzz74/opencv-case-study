"""Command-line entry point for the OpenCV rebar detector."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from rebar_detector.config import load_config
from rebar_detector.pipeline import run_pipeline
from rebar_detector.preprocess import parse_perspective_points
from rebar_detector.visualization import format_text_report


# 直接修改下面的参数后运行 `python main.py`。
PROJECT_DIR = Path(__file__).resolve().parent
# 可选输入图片路径
SELECTED_DATASET_IMAGES = [
    PROJECT_DIR / "Dataset" / "rebar_01.jpg",
    PROJECT_DIR / "Dataset" /  "rebar_02.jpg",
    PROJECT_DIR / "Dataset" /  "rebar_03.jpg",
    PROJECT_DIR / "Dataset" /  "rebar_04.jpg",
    PROJECT_DIR / "Dataset" /  "rebar_05.jpg",
]
# 输入图片：0，1，2，3，4
INPUT_IMAGE_PATH = SELECTED_DATASET_IMAGES[0]
# 可选输出图片路径
OUTPUT_DIR = PROJECT_DIR / "output"
# 指定毫米标定和透视点
MM_PER_PIXEL = None
PERSPECTIVE_POINTS = None
'''
MM_PER_PIXEL = 2.5
PERSPECTIVE_POINTS = "0,60;332,60;332,237;0,237
'''


class _FriendlyArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mm-per-pixel 必须是正数") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("mm-per-pixel 必须是正数")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = _FriendlyArgumentParser(description="钢筋检测与间距分析")
    parser.add_argument(
        "--input",
        default=str(INPUT_IMAGE_PATH),
        help=f"输入图像路径，默认读取 {INPUT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"输出目录，默认写入 {OUTPUT_DIR}",
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置路径")
    parser.add_argument(
        "--mm-per-pixel",
        type=_positive_float,
        default=MM_PER_PIXEL,
        help="标定比例（毫米/像素，必须大于 0）",
    )
    parser.add_argument(
        "--perspective-points",
        default=PERSPECTIVE_POINTS,
        help="透视点：左上;右上;右下;左下，格式 x,y;x,y;x,y;x,y",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the detector CLI and return a conventional process exit code."""
    try:
        arguments = _build_parser().parse_args(argv)
        config = load_config(arguments.config)
        points = parse_perspective_points(arguments.perspective_points)
        report = run_pipeline(
            arguments.input,
            arguments.output,
            config,
            mm_per_pixel=arguments.mm_per_pixel,
            perspective_points=points,
        )
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(format_text_report(report), end="")
    print(f"输出目录：{arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""OpenCV 数据增强课堂案例。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_IMAGE_PATH = PROJECT_DIR / "datsset" / "image" / "001.png"
OUTPUT_DIR = PROJECT_DIR / "output"
SHOW_WINDOW = True
RANDOM_SEED = 42

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def validate_image(image: np.ndarray) -> None:
    """检查图像是否为非空的 uint8 灰度图或彩色图。"""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("图像必须是非空的 NumPy 数组")
    if image.ndim not in (2, 3):
        raise ValueError("图像必须是二维灰度图或三维彩色图")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("图像通道数必须为 1、3 或 4")
    if image.dtype != np.uint8:
        raise ValueError("图像数据类型必须为 uint8")


def read_image(path: str | Path) -> np.ndarray:
    """从字节流解码图像，以支持 Windows 中文路径。"""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图像不存在：{image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"输入图像为空：{image_path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法解码输入图像：{image_path}")
    validate_image(image)
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """编码并保存图像，以支持 Windows 中文路径。"""
    validate_image(image)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"无法编码输出图像：{output_path}")
    output_path.write_bytes(encoded.tobytes())


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}必须是有限数字")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}必须是有限数字")
    return result


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """水平翻转，模拟不同拍摄方向。"""
    validate_image(image)
    return cv2.flip(image, 1)


def rotate_image(image: np.ndarray, angle_degrees: float = 15.0) -> np.ndarray:
    """按顺时针角度旋转，输出尺寸保持不变。"""
    validate_image(image)
    angle = _finite_number(angle_degrees, "旋转角度")
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """按比例调整亮度，并将像素裁剪到 0～255。"""
    validate_image(image)
    brightness_factor = _finite_number(factor, "亮度系数")
    if brightness_factor < 0:
        raise ValueError("亮度系数不能为负数")
    adjusted = image.astype(np.float32) * brightness_factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """使用正方形高斯核模拟相机失焦。"""
    validate_image(image)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError("高斯卷积核大小必须是正奇数")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("高斯卷积核大小必须是正奇数")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    stddev: float = 20.0,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """添加可复现的高斯噪声。"""
    validate_image(image)
    noise_mean = _finite_number(mean, "噪声均值")
    noise_stddev = _finite_number(stddev, "噪声标准差")
    if noise_stddev < 0:
        raise ValueError("噪声标准差不能为负数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("随机种子必须是整数")
    generator = np.random.default_rng(seed)
    noise = generator.normal(noise_mean, noise_stddev, image.shape)
    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def validate_augmented_result(
    name: str, result: np.ndarray, reference: np.ndarray
) -> None:
    """验证增强结果的尺寸、类型和像素范围。"""
    validate_image(reference)
    validate_image(result)
    if result.shape != reference.shape:
        raise ValueError(f"{name}结果尺寸或通道数与原图不一致")
    if int(result.min()) < 0 or int(result.max()) > 255:
        raise ValueError(f"{name}结果像素超出 0～255")


def combined_augmentation(image: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """依次执行旋转、亮度降低、模糊和加噪声。"""
    result = rotate_image(image, angle_degrees=15.0)
    result = adjust_brightness(result, factor=0.55)
    result = gaussian_blur(result, kernel_size=7)
    return add_gaussian_noise(result, mean=0.0, stddev=20.0, seed=seed)


OUTPUT_FILENAMES = {
    "原图": "00_原图.png",
    "水平翻转": "01_水平翻转.png",
    "顺时针旋转 15°": "02_旋转.png",
    "亮度降低": "03_亮度降低.png",
    "亮度提高": "04_亮度提高.png",
    "高斯模糊": "05_高斯模糊.png",
    "高斯噪声": "06_添加高斯噪声.png",
    "多增强组合": "07_多增强组合.png",
}


def build_augmented_images(
    image: np.ndarray, seed: int = RANDOM_SEED
) -> dict[str, np.ndarray]:
    """生成课堂对比所需的全部图像。"""
    validate_image(image)
    images = {
        "原图": image.copy(),
        "水平翻转": horizontal_flip(image),
        "顺时针旋转 15°": rotate_image(image, angle_degrees=15.0),
        "亮度降低": adjust_brightness(image, factor=0.55),
        "亮度提高": adjust_brightness(image, factor=1.45),
        "高斯模糊": gaussian_blur(image, kernel_size=7),
        "高斯噪声": add_gaussian_noise(image, stddev=20.0, seed=seed),
        "多增强组合": combined_augmentation(image, seed=seed),
    }
    for name, result in images.items():
        validate_augmented_result(name, result, image)
    return images


def save_augmented_images(
    images: dict[str, np.ndarray], output_dir: str | Path
) -> list[Path]:
    """按固定的教学顺序保存原图和增强结果。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, filename in OUTPUT_FILENAMES.items():
        if name not in images:
            raise ValueError(f"缺少待保存的图像：{name}")
        path = destination / filename
        write_image(path, images[name])
        paths.append(path)
    return paths


def _display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def create_comparison_figure(
    images: dict[str, np.ndarray],
    output_path: str | Path,
    show: bool = True,
) -> Path:
    """生成并保存 2×4 数据增强效果对比图。"""
    if list(images) != list(OUTPUT_FILENAMES):
        raise ValueError("对比图的图像名称或顺序不正确")
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, image) in zip(axes.flat, images.items()):
        display = _display_image(image)
        axis.imshow(display, cmap="gray" if display.ndim == 2 else None)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    figure.suptitle("施工现场图像数据增强效果对比", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return destination


def run_demo(
    input_path: str | Path = INPUT_IMAGE_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    show: bool = SHOW_WINDOW,
) -> list[Path]:
    """运行完整教学案例并返回生成文件路径。"""
    image = read_image(input_path)
    images = build_augmented_images(image, seed=RANDOM_SEED)
    generated = save_augmented_images(images, output_dir)
    comparison_path = create_comparison_figure(
        images,
        Path(output_dir) / "数据增强效果对比.png",
        show=show,
    )
    generated.append(comparison_path)
    return generated


def main() -> int:
    """运行课堂案例并返回程序退出码。"""
    print("\n" + "=" * 60)
    print("OpenCV 数据增强课堂教学案例")
    print("=" * 60)
    print(f"输入图像：{INPUT_IMAGE_PATH}")
    try:
        generated = run_demo(show=SHOW_WINDOW)
    except (FileNotFoundError, ValueError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print("\n已生成以下结果：")
    for path in generated:
        print(f"  - {path}")
    print("\n提示：可修改 main.py 顶部参数和增强函数默认值进行课堂实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

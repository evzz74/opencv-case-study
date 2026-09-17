# 数据增强课堂教学案例实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 使用 `数据增强/datsset/image/001.png` 构建一个可直接运行的中文课堂案例，生成水平翻转、旋转、亮度变化、高斯模糊、高斯噪声、多增强组合和 2×4 效果对比图。

**Architecture:** 在 `数据增强/main.py` 中用职责独立的函数实现图像校验、中文路径读写、单项增强、组合增强、结果保存和 Matplotlib 可视化。使用标准库 `unittest` 对函数约束及端到端输出进行回归验证，README 负责原理、工程场景、运行方法和课堂练习说明。

**Tech Stack:** Python 3.10+、OpenCV、NumPy、Matplotlib、标准库 unittest

## Global Constraints

- 所有新增文本文件使用 UTF-8 编码。
- 保留输入图片路径 `数据增强/datsset/image/001.png`，不移动或重命名现有图片。
- 默认增强参数固定为：顺时针旋转 15°、亮度系数 0.55/1.45、7×7 高斯核、高斯噪声均值 0/标准差 20/随机种子 42。
- 多增强组合顺序固定为：旋转 → 亮度降低 → 高斯模糊 → 高斯噪声。
- 所有增强结果必须与原图尺寸、通道数和 `uint8` 类型一致，像素值保持在 `[0, 255]`。
- Windows 中文路径必须通过 `np.fromfile`/`cv2.imdecode` 和 `cv2.imencode`/`Path.write_bytes` 支持。
- 成功运行返回退出码 0；预期输入、参数、编码和写入错误使用中文信息并返回非零退出码。
- 按用户选择，不创建 Git 提交。

---

## 文件结构

**创建：**

- `数据增强/main.py`：增强函数、运行流水线、结果保存和对比图。
- `数据增强/tests/test_main.py`：单文件标准库回归测试，不增加测试依赖。
- `数据增强/requirements.txt`：OpenCV、NumPy、Matplotlib 运行依赖。
- `数据增强/README.md`：课堂讲义和运行说明。

**运行后生成：**

- `数据增强/output/00_原图.png`
- `数据增强/output/01_水平翻转.png`
- `数据增强/output/02_旋转.png`
- `数据增强/output/03_亮度降低.png`
- `数据增强/output/04_亮度提高.png`
- `数据增强/output/05_高斯模糊.png`
- `数据增强/output/06_添加高斯噪声.png`
- `数据增强/output/07_多增强组合.png`
- `数据增强/output/数据增强效果对比.png`

---

### Task 1: 实现图像读写和基础增强函数

**Files:**
- Create: `数据增强/main.py`
- Create: `数据增强/tests/test_main.py`

**Interfaces:**
- Consumes: OpenCV `cv2`、NumPy `np`、`pathlib.Path`。
- Produces:
  - `validate_image(image: np.ndarray) -> None`
  - `read_image(path: str | Path) -> np.ndarray`
  - `write_image(path: str | Path, image: np.ndarray) -> None`
  - `horizontal_flip(image: np.ndarray) -> np.ndarray`
  - `rotate_image(image: np.ndarray, angle_degrees: float = 15.0) -> np.ndarray`
  - `adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray`
  - `gaussian_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray`
  - `add_gaussian_noise(image: np.ndarray, mean: float = 0.0, stddev: float = 20.0, seed: int = 42) -> np.ndarray`
  - `validate_augmented_result(name: str, result: np.ndarray, reference: np.ndarray) -> None`
  - `combined_augmentation(image: np.ndarray, seed: int = 42) -> np.ndarray`

- [x] **Step 1: 创建基础增强的失败测试**

创建 `数据增强/tests/test_main.py`，使用动态路径导入，避免中文目录不是 Python 包时发生导入问题：

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_DIR / "main.py"
SPEC = importlib.util.spec_from_file_location("data_augmentation_main", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


class AugmentationFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(6 * 8 * 3, dtype=np.uint8)
        self.image = values.reshape(6, 8, 3)

    def test_horizontal_flip_matches_numpy_reference(self) -> None:
        result = main.horizontal_flip(self.image)
        np.testing.assert_array_equal(result, self.image[:, ::-1])

    def test_rotation_preserves_shape_and_type(self) -> None:
        result = main.rotate_image(self.image, angle_degrees=15.0)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_brightness_clips_to_uint8_range(self) -> None:
        bright = main.adjust_brightness(
            np.full((2, 2, 3), 240, dtype=np.uint8), 1.45
        )
        dark = main.adjust_brightness(
            np.full((2, 2, 3), 100, dtype=np.uint8), 0.55
        )
        self.assertTrue(np.all(bright == 255))
        self.assertTrue(np.all(dark == 55))
        self.assertEqual(bright.dtype, np.uint8)

    def test_gaussian_blur_rejects_even_kernel(self) -> None:
        with self.assertRaisesRegex(ValueError, "正奇数"):
            main.gaussian_blur(self.image, kernel_size=6)

    def test_noise_is_reproducible_for_same_seed(self) -> None:
        first = main.add_gaussian_noise(self.image, seed=42)
        second = main.add_gaussian_noise(self.image, seed=42)
        other = main.add_gaussian_noise(self.image, seed=43)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(first.shape, self.image.shape)
        self.assertEqual(first.dtype, np.uint8)

    def test_combined_augmentation_preserves_contract(self) -> None:
        result = main.combined_augmentation(self.image, seed=42)
        main.validate_augmented_result("多增强组合", result, self.image)

    def test_unicode_image_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "中文图片.png"
            main.write_image(output_path, self.image)
            decoded = main.read_image(output_path)
            np.testing.assert_array_equal(decoded, self.image)

    def test_invalid_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "图像"):
            main.validate_image(np.array([], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行测试并确认初始失败**

Run:

```bash
python -m unittest discover -s "数据增强/tests" -v
```

Expected: 导入 `数据增强/main.py` 失败，或测试报告增强函数尚未定义。

- [x] **Step 3: 实现配置、校验和中文路径读写**

在 `数据增强/main.py` 顶部创建配置和基础函数：

```python
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
```

- [x] **Step 4: 实现六类增强和结果契约校验**

继续在 `数据增强/main.py` 中添加：

```python
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
```

- [x] **Step 5: 运行基础函数测试**

Run:

```bash
python -m unittest discover -s "数据增强/tests" -v
```

Expected: 8 tests PASS；不得出现导入错误、中文路径错误或随机性失败。

---

### Task 2: 实现完整增强流水线和 2×4 对比图

**Files:**
- Modify: `数据增强/main.py`
- Modify: `数据增强/tests/test_main.py`

**Interfaces:**
- Consumes: Task 1 提供的全部图像读写、增强和校验函数。
- Produces:
  - `build_augmented_images(image: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]`
  - `save_augmented_images(images: dict[str, np.ndarray], output_dir: str | Path) -> list[Path]`
  - `create_comparison_figure(images: dict[str, np.ndarray], output_path: str | Path, show: bool = True) -> Path`
  - `run_demo(input_path: str | Path = INPUT_IMAGE_PATH, output_dir: str | Path = OUTPUT_DIR, show: bool = SHOW_WINDOW) -> list[Path]`
  - `main() -> int`

- [x] **Step 1: 添加端到端失败测试**

在 `数据增强/tests/test_main.py` 中追加：

```python
class PipelineTests(unittest.TestCase):
    def test_build_augmented_images_has_expected_order_and_contract(self) -> None:
        image = np.full((20, 30, 3), 120, dtype=np.uint8)
        results = main.build_augmented_images(image, seed=42)
        self.assertEqual(
            list(results),
            [
                "原图",
                "水平翻转",
                "顺时针旋转 15°",
                "亮度降低",
                "亮度提高",
                "高斯模糊",
                "高斯噪声",
                "多增强组合",
            ],
        )
        for name, result in results.items():
            main.validate_augmented_result(name, result, image)

    def test_run_demo_generates_all_expected_png_files(self) -> None:
        source = PROJECT_DIR / "datsset" / "image" / "001.png"
        with tempfile.TemporaryDirectory() as directory:
            generated = main.run_demo(source, directory, show=False)
            expected_names = {
                "00_原图.png",
                "01_水平翻转.png",
                "02_旋转.png",
                "03_亮度降低.png",
                "04_亮度提高.png",
                "05_高斯模糊.png",
                "06_添加高斯噪声.png",
                "07_多增强组合.png",
                "数据增强效果对比.png",
            }
            self.assertEqual({path.name for path in generated}, expected_names)
            for path in generated:
                self.assertTrue(path.is_file(), path)
                data = np.fromfile(str(path), dtype=np.uint8)
                decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
                self.assertIsNotNone(decoded, path)
```

- [x] **Step 2: 运行新增测试并确认失败**

Run:

```bash
python -m unittest discover -s "数据增强/tests" -v
```

Expected: 基础函数测试通过；流水线测试因 `build_augmented_images` 或 `run_demo` 尚未定义而失败。

- [x] **Step 3: 实现结果集合和单图保存**

在 `数据增强/main.py` 中添加固定输出映射和流水线：

```python
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
```

- [x] **Step 4: 实现 Matplotlib 对比图**

继续添加颜色转换和绘图函数：

```python
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
```

- [x] **Step 5: 实现运行入口和中文终端输出**

在文件末尾添加：

```python
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
```

- [x] **Step 6: 运行全部自动化测试**

Run:

```bash
python -m unittest discover -s "数据增强/tests" -v
```

Expected: 10 tests PASS；临时目录中生成 9 个可解码 PNG 文件。

- [x] **Step 7: 运行无窗口端到端验证**

Run:

```bash
python -c "import importlib.util, pathlib; p=pathlib.Path('数据增强/main.py'); s=importlib.util.spec_from_file_location('demo', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); paths=m.run_demo(show=False); print(len(paths)); print('\n'.join(str(x) for x in paths))"
```

Expected:

```text
9
数据增强/output/00_原图.png
...
数据增强/output/数据增强效果对比.png
```

路径可能显示为绝对路径，但文件数量必须为 9。

---

### Task 3: 编写依赖和课堂讲义并完成最终验证

**Files:**
- Create: `数据增强/requirements.txt`
- Create: `数据增强/README.md`
- Verify: `数据增强/output/*.png`

**Interfaces:**
- Consumes: Task 2 提供的 `python main.py` 运行入口和全部输出文件名。
- Produces: 可安装依赖清单、可独立使用的课堂讲义、与文档一致的最终运行结果。

- [x] **Step 1: 创建运行依赖文件**

创建 `数据增强/requirements.txt`：

```text
opencv-python>=4.8
numpy>=1.24
matplotlib>=3.7
```

- [x] **Step 2: 创建 README 的案例目标和工程场景部分**

创建 `数据增强/README.md`，开头必须包含：

```markdown
# 图像数据增强课堂教学案例

本案例使用施工现场图片，通过 OpenCV、NumPy 和 Matplotlib 展示常见图像数据增强方法。

## 教学目标

- 区分几何增强和像素增强；
- 理解翻转、旋转、亮度、模糊和噪声的基本实现；
- 掌握像素类型转换、范围裁剪和随机种子；
- 理解多种增强操作组成的数据增强流水线。

## 数据增强与工程场景

| 数据增强 | 模拟的工程场景 | 难度 |
| --- | --- | ---: |
| 水平翻转 | 不同拍摄方向 | ★ |
| 旋转 | 相机倾斜、手持拍摄 | ★ |
| 亮度变化 | 阴天、强光、弱光 | ★ |
| 高斯模糊 | 相机失焦、轻微运动模糊 | ★ |
| 添加噪声 | 低质量摄像头、复杂环境 | ★★ |
| 多增强组合 | 多种不利拍摄条件同时出现 | ★★ |
```

- [x] **Step 3: 补充安装、运行、流程和输出说明**

README 必须给出可复制命令和准确路径：

````markdown
## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行

在 `数据增强` 目录执行：

```bash
python main.py
```

输入图片：

```text
datsset/image/001.png
```

程序会自动创建 `output/`，保存 8 张单图和 1 张 2×4 对比总图。

## 处理流程

```text
原图
 │
 ├── 水平翻转
 ├── 顺时针旋转 15°
 ├── 亮度降低 / 亮度提高
 ├── 高斯模糊
 ├── 高斯噪声
 └── 旋转 → 亮度降低 → 模糊 → 噪声
                  │
                  ↓
             保存单图和总览图
```
````

随后逐项列出 9 个输出文件名，必须与 `OUTPUT_FILENAMES` 和 `create_comparison_figure()` 一致。

- [x] **Step 4: 补充原理、参数实验和课堂思考题**

README 对每种增强分别解释：

- `cv2.flip(image, 1)` 中参数 `1` 的含义；
- `cv2.getRotationMatrix2D()` 的角度方向，代码用 `-15` 实现顺时针 15°；
- 亮度计算为何先转 `float32`，再 `np.clip(..., 0, 255)` 和转回 `uint8`；
- 高斯卷积核必须是正奇数，增大卷积核会增强模糊；
- 高斯噪声均值、标准差和随机种子的含义；
- 多增强组合的执行顺序为什么会影响结果；
- 训练集可随机增强，验证集和测试集通常保持确定性预处理。

加入以下可操作实验：

```markdown
## 参数调节实验

1. 将旋转角度从 `15.0` 改为 `30.0`，观察裁剪和黑边变化。
2. 将弱光系数从 `0.55` 改为 `0.30`，比较细节损失。
3. 将高斯核从 `7` 改为 `3` 或 `11`，比较模糊程度。
4. 将噪声标准差从 `20.0` 改为 `5.0` 和 `40.0`，比较图像质量。
5. 交换“模糊”和“加噪声”的顺序，比较组合增强结果。

## 课堂思考题

1. 为什么旋转后会出现黑边？
2. 为什么亮度变化和添加噪声前要转换数据类型？
3. 为什么像素计算后必须裁剪到 `[0, 255]`？
4. 为什么固定随机种子有助于课堂复现？
5. 模糊后加噪声与加噪声后模糊的结果是否相同？
6. 为什么通常只对训练集执行随机数据增强？
```

- [x] **Step 5: 检查 Python 语法和全部回归测试**

Run:

```bash
python -m py_compile "数据增强/main.py" "数据增强/tests/test_main.py"
python -m unittest discover -s "数据增强/tests" -v
```

Expected: 编译无输出；10 tests PASS。

- [x] **Step 6: 清理旧输出并执行最终端到端运行**

先查看 `数据增强/output/`；仅删除本案例此前生成、且文件名与本计划明确列出的 9 个文件完全一致的输出。不得删除未知的用户文件。随后运行：

```bash
python -c "import importlib.util, pathlib; p=pathlib.Path('数据增强/main.py'); s=importlib.util.spec_from_file_location('demo', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); raise SystemExit(0 if len(m.run_demo(show=False)) == 9 else 1)"
```

Expected: 退出码 0，并重新生成 9 个文件。

- [x] **Step 7: 校验输出尺寸、类型和可解码性**

Run:

```bash
python -c "import cv2, numpy as np, pathlib; d=pathlib.Path('数据增强/output'); files=sorted(d.glob('*.png')); assert len(files)==9, len(files); decoded=[]; [(lambda p: decoded.append((p, cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR))))(p) for p in files]; assert all(img is not None and img.size for _, img in decoded); source=cv2.imdecode(np.fromfile('数据增强/datsset/image/001.png', dtype=np.uint8), cv2.IMREAD_COLOR); singles=[img for p,img in decoded if p.name!='数据增强效果对比.png']; assert all(img.shape==source.shape and img.dtype==np.uint8 for img in singles); print('verified', len(files), 'png files')"
```

Expected:

```text
verified 9 png files
```

- [x] **Step 8: 人工检查课堂展示质量**

打开 `数据增强/output/数据增强效果对比.png`，确认：

- 2×4 布局完整；
- 八个中文标题与图像对应；
- 原图颜色正常，不出现 BGR/RGB 颜色颠倒；
- 水平翻转方向明显；
- 旋转为顺时针且黑边合理；
- 亮度降低和提高均可辨认；
- 模糊、噪声和组合增强效果清晰但不至于完全破坏主体；
- 标题不重叠、不截断。

如仅出现 Matplotlib 中文字体警告但图片标题正常，可记录警告；如果标题显示为方框，应调整 `font.sans-serif` 候选字体后重新生成。

- [x] **Step 9: 检查文档与实现一致性**

逐项核对：

- README 的输入路径与 `INPUT_IMAGE_PATH` 一致；
- README 的 9 个输出文件与实际目录一致；
- README 参数值与函数默认值一致；
- README 的组合顺序与 `combined_augmentation()` 一致；
- `requirements.txt` 覆盖 `main.py` 的三个第三方导入；
- `git diff --check` 不报告空白错误。

Run:

```bash
git diff --check
```

Expected: 无输出。

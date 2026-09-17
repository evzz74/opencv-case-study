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


if __name__ == "__main__":
    unittest.main()

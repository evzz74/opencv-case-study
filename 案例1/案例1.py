"""
风机叶片缺陷检测
支持YOLO格式标注的可视化和统计分析
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BladeDefectDetector:
    """风机叶片缺陷检测器"""

    def __init__(self, images_dir, labels_dir):
        """
        初始化检测器

        Args:
            images_dir: 图像文件夹路径
            labels_dir: 标注文件夹路径
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = {0: "Crack"}  # 标签名称设置
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    def load_label(self, label_path):
        """
        加载YOLO格式的标注文件

        Args:
            label_path: 标注文件路径

        Returns:
            list: 标注列表，每个标注为 [class_id, center_x, center_y, width, height]
        """
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        annotations.append([class_id] + coords)
        return annotations

    def yolo_to_bbox(self, img_width, img_height, yolo_coords):
        """
        将YOLO格式坐标转换为边界框坐标

        Args:
            img_width: 图像宽度
            img_height: 图像高度
            yolo_coords: YOLO格式坐标 [center_x, center_y, width, height]

        Returns:
            tuple: (x1, y1, x2, y2) 边界框坐标
        """
        center_x, center_y, width, height = yolo_coords

        # 转换为像素坐标
        center_x *= img_width
        center_y *= img_height
        width *= img_width
        height *= img_height

        # 计算边界框左上角和右下角坐标
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        return x1, y1, x2, y2

    def draw_detections(self, image, annotations):
        """
        在图像上绘制检测框

        Args:
            image: 输入图像
            annotations: 标注列表

        Returns:
            绘制了检测框的图像
        """
        img_height, img_width = image.shape[:2]
        result_img = image.copy()

        for ann in annotations:
            class_id = ann[0]
            yolo_coords = ann[1:]

            # 转换坐标
            x1, y1, x2, y2 = self.yolo_to_bbox(img_width, img_height, yolo_coords)

            # 选择颜色
            color = self.colors[class_id % len(self.colors)]

            # 绘制边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = self.class_names.get(class_id, f"类别{class_id}")
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 绘制标签背景
            cv2.rectangle(result_img,
                         (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1),
                         color, -1)

            # 绘制标签文字
            cv2.putText(result_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result_img

    def detect_single_image(self, image_name, show=True, save=False, output_dir=None):
        """
        对单张图像进行检测

        Args:
            image_name: 图像文件名
            show: 是否显示结果
            save: 是否保存结果
            output_dir: 输出目录

        Returns:
            检测结果图像和标注列表
        """
        # 加载图像
        img_path = self.images_dir / image_name
        if not img_path.exists():
            print(f"图像不存在: {img_path}")
            return None, None

        # 使用numpy读取以支持中文路径
        try:
            with open(str(img_path), 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                print(f"无法解码图像: {img_path}")
                return None, None
        except Exception as e:
            print(f"读取图像失败 {img_path}: {e}")
            return None, None

        # 加载标注
        label_name = img_path.stem + '.txt'
        label_path = self.labels_dir / label_name
        annotations = self.load_label(label_path)

        # 绘制检测框
        result_img = self.draw_detections(image, annotations)

        # 显示结果
        if show:
            # 转换为RGB显示
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(result_rgb)
            plt.title(f'检测结果 - {image_name}\n检测到 {len(annotations)} 个缺陷')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        # 保存结果
        if save and output_dir:
            output_path = Path(output_dir) / f"detected_{image_name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 使用imencode支持中文路径
            is_success, encoded_img = cv2.imencode('.jpg', result_img)
            if is_success:
                with open(str(output_path), 'wb') as f:
                    f.write(encoded_img.tobytes())
                print(f"结果已保存到: {output_path}")
            else:
                print(f"保存图像失败: {output_path}")

        return result_img, annotations

    def detect_all_images(self, max_display=5, save=False, output_dir="output"):
        """
        检测所有图像

        Args:
            max_display: 最多显示的图像数量
            save: 是否保存结果
            output_dir: 输出目录

        Returns:
            统计信息字典
        """
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))

        if not image_files:
            print("未找到图像文件")
            return None

        print(f"共找到 {len(image_files)} 张图像")

        # 统计信息
        stats = {
            'total_images': len(image_files),
            'total_defects': 0,
            'images_with_defects': 0,
            'defects_per_image': []
        }

        # 处理每张图像
        for idx, img_path in enumerate(image_files):
            image_name = img_path.name

            # 只显示前几张
            show_flag = idx < max_display

            result_img, annotations = self.detect_single_image(
                image_name,
                show=show_flag,
                save=save,
                output_dir=output_dir
            )

            if annotations is not None:
                num_defects = len(annotations)
                stats['total_defects'] += num_defects
                stats['defects_per_image'].append(num_defects)
                if num_defects > 0:
                    stats['images_with_defects'] += 1

                print(f"[{idx+1}/{len(image_files)}] {image_name}: {num_defects} 个缺陷")

        # 计算统计数据
        if stats['defects_per_image']:
            stats['avg_defects'] = np.mean(stats['defects_per_image'])
            stats['max_defects'] = np.max(stats['defects_per_image'])
            stats['min_defects'] = np.min(stats['defects_per_image'])

        return stats

    def print_statistics(self, stats):
        """
        打印统计信息

        Args:
            stats: 统计信息字典
        """
        if stats is None:
            return

        print("\n" + "="*50)
        print("数据集统计信息")
        print("="*50)
        print(f"总图像数量: {stats['total_images']}")
        print(f"有缺陷的图像: {stats['images_with_defects']}")
        print(f"总缺陷数量: {stats['total_defects']}")

        if 'avg_defects' in stats:
            print(f"平均每张图像缺陷数: {stats['avg_defects']:.2f}")
            print(f"单张图像最多缺陷数: {stats['max_defects']}")
            print(f"单张图像最少缺陷数: {stats['min_defects']}")
        print("="*50)


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    output_dir = base_dir / "output"

    # 创建检测器
    detector = BladeDefectDetector(images_dir, labels_dir)

    print("风机叶片缺陷检测")
    print("="*50)

    # 检测所有图像
    stats = detector.detect_all_images(
        max_display=3,  # 只显示前3张图像
        save=True,      # 保存检测结果
        output_dir=output_dir
    )

    # 打印统计信息
    detector.print_statistics(stats)

if __name__ == "__main__":
    main()

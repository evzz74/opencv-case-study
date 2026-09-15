# 钢筋数量、位置与间距检测

基于 Python + OpenCV 的传统视觉教学项目，使用：

```text
灰度 → 高斯滤波 → Canny → HoughLinesP
→ 直线方向聚类与合并 → 钢筋数量/位置/方向/间距/异常分析
```

## 1. 安装

在项目目录执行：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. 数据集

项目数据集位于：

```text
.\Dataset
```

项目整理后只保留以下 5 张图片：

```text
Dataset\selected\rebar_01.jpg
Dataset\selected\rebar_02.jpg
Dataset\selected\rebar_03.jpg
Dataset\selected\rebar_04.jpg
Dataset\selected\rebar_05.jpg
```

图片已重新命名，列表和默认图片已在 `main.py` 中配置：

```python
SELECTED_DATASET_IMAGES = [...]
INPUT_IMAGE_PATH = SELECTED_DATASET_IMAGES[0]
```

## 3. 运行方式

本项目推荐直接修改 `main.py` 顶部的运行参数，然后执行：

```bash
python main.py
```

需要修改的参数：

```python
INPUT_IMAGE_PATH = PROJECT_DIR / "Dataset" / "rebar_02.jpg"
OUTPUT_DIR = PROJECT_DIR / "output"
MM_PER_PIXEL = None
PERSPECTIVE_POINTS = None
```

例如指定毫米标定和透视点：

```python
MM_PER_PIXEL = 2.5
PERSPECTIVE_POINTS = "0,60;332,60;332,237;0,237"
```

透视点顺序必须是：

```text
左上;右上;右下;左下
```

普通照片没有物理尺度时，`MM_PER_PIXEL` 应保持为 `None`，程序会用 `px` 输出间距；不能直接把像素值当作毫米。

命令行参数仍然可以临时覆盖配置：

```bash
python main.py --input "Dataset\rebar_03.jpg" --output output-rebar-03
python main.py --input "Dataset\rebar_05.jpg" --output output-rebar-05
```

## 4. 输出结果

输出目录默认是 `output/`：

```text
output/
├── annotated.png       # 原图/矫正图上的钢筋中心线、编号和间距箭头
├── report.json         # 结构化检测结果
├── report.txt          # 中文检测报告
└── debug/
    ├── 00_rectified.png    # 启用透视矫正时生成
    ├── 01_gray.png         # 灰度图
    ├── 02_blur.png         # 高斯滤波图
    ├── 03_edges.png        # Canny 边缘图
    ├── 04_hough_raw.png    # Hough 原始线段
    └── 05_merged_lines.png # 合并后的钢筋中心线
```

报告按方向分别给出：

- 钢筋数量；
- 每根钢筋的位置和方向角；
- 相邻钢筋像素间距或毫米间距；
- 平均、最大、最小和中位间距；
- 相对中位数偏差超过 20% 的明显异常间距。

## 5. 参数调节

默认参数在 `config/default.yaml`，已按上述 5 张 1500×900 左右的现场训练图片做过基础调节：

- Hough 阈值和最短线段长度比小尺寸示例图更高；
- 方向组和中心线合并要求更强的支持；
- 其他图片仍可能需要针对现场光照和透视单独微调。

| 参数 | 作用 |
| --- | --- |
| `preprocess.canny_low/high` | Canny 边缘阈值 |
| `hough.threshold` | Hough 检测敏感度 |
| `hough.min_line_length` | 最短线段长度 |
| `hough.max_line_gap` | 断裂线段连接间隔 |
| `grouping.angle_tolerance_degrees` | 方向聚类容差 |
| `merging.normal_cluster_distance_px` | 碎线合并距离 |
| `merging.duplicate_center_distance_px` | 双边缘合并距离 |
| `measurement.anomaly_ratio` | 异常间距阈值，默认 0.20 |

背景直线太多时提高 `hough.threshold`；钢筋断裂严重时适当提高 `max_line_gap`；同一根粗钢筋被识别为两根时提高 `duplicate_center_distance_px`。

## 6. 运行测试

```bash
python -m pytest -q
```

本项目仅使用传统 OpenCV 方法，适合用于课堂演示；严重遮挡、弯曲、反光或透视很强的图片需要人工检查和参数调整。
# -*- coding: utf-8 -*-
import cv2  # OpenCV库 - 计算机视觉核心库
import numpy as np  # NumPy - 数组运算和数值计算
import os
from pathlib import Path  # 路径操作
import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端 
import matplotlib.pyplot as plt  # 绘图库
from skimage.morphology import skeletonize  # 骨架提取算法
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



def analyze_crack_with_opencv(image_path):
    """
    使用OpenCV灰度检测方法分析裂缝

    完整的图像处理流程:
    预处理 → 阈值分割 → 形态学操作 → 连通域分析 → 特征提取

    参数:
        image_path: 图像文件路径

    返回:
        dict: 包含分析结果的字典，包括孔隙率、裂缝长度等信息
    """
    # ==================== 步骤1: 图像读取 ====================
    # cv2.imread() 在 Windows 上无法处理中文路径
    #使用 numpy 读取文件字节流，然后用 cv2.imdecode 解码
    # cv2.imread() 读取图像，默认为BGR格式
    img_array = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法解码图像: {image_path}")
        return None


    # ==================== 步骤2: 颜色空间转换 ====================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_gray = gray.copy()  # 保存原始灰度图，用于后续灰度值分析

    # ==================== 步骤3: 图像滤波去噪 ====================
    # 3.1 中值滤波 (Median Filter)
    # 去除椒盐噪声 (图像中的黑白点)
    denoised = cv2.medianBlur(gray, 3)

    # 3.2 双边滤波 (Bilateral Filter)
    # 平滑图像同时保持边缘清晰
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)

    # ==================== 步骤4: 阈值分割 ====================
    # 【教学重点】两种阈值方法的对比与组合

    # 4.1 OTSU自动阈值分割
    # 通过统计方法自动找到最优阈值，将图像分为前景和背景
    # 【参数说明】
    #   - bilateral: 输入的灰度图
    #   - 0: 阈值参数 (使用OTSU时该值会被忽略，自动计算)
    #   - 255: 最大值 (前景像素会被设为255)
    #   - cv2.THRESH_BINARY_INV: 反向二值化 (暗的部分变白，亮的部分变黑)
    #   - cv2.THRESH_OTSU: 使用OTSU算法自动计算阈值
    # 【返回值】
    #   - otsu_thresh: 自动计算的阈值
    #   - otsu_binary: 二值化后的图像
    otsu_thresh, otsu_binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4.2 自适应阈值分割
    # 根据每个像素周围的局部区域计算阈值，适应光照变化
    # 【参数说明】
    #   - bilateral: 输入的灰度图
    #   - 255: 最大值
    #   - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 使用高斯加权计算阈值
    #   - cv2.THRESH_BINARY_INV: 反向二值化
    #   - 15: blockSize - 局部区域大小 (必须是奇数)
    #   - 3: C - 从计算出的均值中减去的常数，用于微调阈值
    adaptive_binary = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 3)

    # 4.3 组合两种方法 - 取交集
    combined_binary = cv2.bitwise_and(otsu_binary, adaptive_binary)

    # ==================== 步骤5: 形态学操作 ====================
    # 5.1 定义结构元素 (Kernel / Structuring Element)
    kernel_small = np.ones((2, 2), np.uint8)  # 小核，用于精细操作
    kernel = np.ones((3, 3), np.uint8)        # 标准核，用于常规操作

    # 5.2 开运算 (Opening = 先腐蚀后膨胀)
    # 去除小的白色噪声点
    opened = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 5.3 闭运算 (Closing = 先膨胀后腐蚀)
    # 填充小的黑色孔洞，连接断裂的裂缝
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5.4 膨胀操作 (Dilation)
    # 轻微扩大裂缝区域
    dilated = cv2.dilate(closed, kernel_small, iterations=1)

    # ==================== 步骤6: 连通域分析 ====================
    # 6.1 连通域检测
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # 6.2 动态计算最小面积阈值
    min_area = max(30, int(img.shape[0] * img.shape[1] * 0.0001))

    # 6.3 创建裂缝掩码和过滤连通域
    crack_mask = np.zeros_like(dilated)  # 创建空白掩码
    valid_components = []  # 存储有效连通域的ID和面积

    # 计算整体平均灰度，用于判断区域是否为裂缝
    overall_mean = np.mean(original_gray)

    # 6.4 遍历每个连通域进行过滤
    for i in range(1, num_labels):
        # 获取当前连通域的面积
        area = stats[i, cv2.CC_STAT_AREA]  # cv2.CC_STAT_AREA = 4

        # 条件1: 面积过滤 - 面积必须大于最小阈值
        if area >= min_area:
            # 创建当前连通域的掩码
            component_mask = (labels == i)

            # 计算当前连通域的平均灰度值
            mean_gray = np.mean(original_gray[component_mask])

            # 条件2: 灰度过滤 - 比背景暗至少5个灰度值
            if mean_gray < overall_mean - 5:
                # 保留该连通域
                crack_mask[labels == i] = 255
                valid_components.append((i, area))

    # ==================== 步骤7: 骨架提取 ====================
    # 7.1 提取骨架
    # 【注意】
    #   - 输入必须是0-1的二值图，所以除以255
    #   - 输出是bool类型，需要转换回uint8并乘以255
    skeleton = skeletonize(crack_mask // 255).astype(np.uint8) * 255

    # 7.2 对骨架进行连通域分析
    # 找出每一条独立的骨架线(对应一条裂缝)
    skel_labels, skel_labels_img, skel_stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

    # 7.3 查找最长的裂缝
    # 遍历每条骨架，计算对应的轮廓长度
    max_length = 0
    longest_crack_mask = np.zeros_like(crack_mask)

    # 遍历每个骨架连通域
    for label in range(1, skel_labels):
        # 提取当前骨架
        skel_component = (skel_labels_img == label).astype(np.uint8) * 255
        dilated_skel = cv2.dilate(skel_component, kernel, iterations=2)
        # 与原始裂缝掩码相交，得到该骨架对应的完整裂缝
        crack_region = cv2.bitwise_and(crack_mask, dilated_skel)

        # ==================== 步骤8: 轮廓检测与长度计算 ====================
        # 8.1 查找轮廓
        contours, _ = cv2.findContours(crack_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 8.2 计算轮廓长度
        for contour in contours:
            length = cv2.arcLength(contour, False)

            # 更新最长裂缝
            if length > max_length:
                max_length = length
                longest_crack_mask = crack_region.copy()

    # 8.3 备用方案 - 如果骨架提取失败
    if np.sum(longest_crack_mask) == 0 and len(valid_components) > 0:
        # 找到面积最大的连通域
        largest = max(valid_components, key=lambda x: x[1])  # x[1]是面积
        longest_crack_mask[labels == largest[0]] = 255

        # 估算长度 = √面积 × 2 (粗略估计)
        max_length = np.sqrt(largest[1]) * 2

    # ==================== 步骤9: 计算孔隙率 ====================
    # 孔隙率 = 裂缝像素数 / 总像素数 × 100%
    total_pixels = img.shape[0] * img.shape[1]  # 图像总像素数
    crack_pixels = np.sum(crack_mask > 0)  # 裂缝像素数
    porosity = (crack_pixels / total_pixels) * 100  # 百分比形式

    # ==================== 步骤10: 创建可视化 ====================
    # 创建2行4列的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # ========== 第一行: 原始图像和预处理步骤 ==========

    # 10.1 显示原始图像
    # 【注意】OpenCV读取的是BGR格式，需要转换为RGB才能正确显示
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')  # 关闭坐标轴

    # 10.2 显示灰度图
    axes[0, 1].imshow(gray, cmap='gray')  # cmap='gray'指定灰度色图
    axes[0, 1].set_title(f'灰度图\n均值: {overall_mean:.1f}')
    axes[0, 1].axis('off')

    # 10.3 显示OTSU阈值结果
    axes[0, 2].imshow(otsu_binary, cmap='gray')
    axes[0, 2].set_title(f'OTSU阈值\n阈值: {otsu_thresh:.0f}')
    axes[0, 2].axis('off')

    # 10.4 显示自适应阈值结果
    axes[0, 3].imshow(adaptive_binary, cmap='gray')
    axes[0, 3].set_title('自适应阈值')
    axes[0, 3].axis('off')

    # ========== 第二行: 检测结果 ==========

    # 10.5 显示检测到的所有裂缝
    axes[1, 0].imshow(crack_mask, cmap='gray')
    axes[1, 0].set_title(f'检测到的裂缝\n孔隙率: {porosity:.2f}%')
    axes[1, 0].axis('off')

    # 10.6 显示骨架
    axes[1, 1].imshow(skeleton, cmap='gray')
    axes[1, 1].set_title('裂缝骨架')
    axes[1, 1].axis('off')

    # 10.7 显示最长裂缝
    axes[1, 2].imshow(longest_crack_mask, cmap='gray')
    axes[1, 2].set_title(f'最长裂缝\n长度: {max_length:.0f} px')
    axes[1, 2].axis('off')

    # 10.8 显示叠加结果
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[crack_mask > 0] = [255, 0, 0]  # BGR格式 - 蓝色表示所有裂缝
    color_mask[longest_crack_mask > 0] = [0, 0, 255]  # BGR格式 - 红色表示最长裂缝

    # cv2.addWeighted() - 图像加权混合
    result = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    axes[1, 3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('综合结果\n蓝:所有 红:最长')
    axes[1, 3].axis('off')

    # 10.9 添加总标题和布局调整
    plt.suptitle(f'OpenCV裂缝检测分析 - {Path(image_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()  # 自动调整子图间距

    # 10.10 保存结果图像
    # 【路径】保存到脚本所在目录下的outputs文件夹
    script_dir = Path(__file__).parent  # 获取脚本所在目录
    output_dir = script_dir / "outputs"  # 输出文件夹路径
    output_dir.mkdir(exist_ok=True)  # 如果文件夹不存在则创建
    output_path = output_dir / f"{Path(image_path).stem}_opencv_analysis.png"

    # 【参数说明】
    #   - dpi=150: 分辨率，决定图像清晰度
    #   - bbox_inches='tight': 紧凑裁剪，去除多余空白
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()  # 显示图形窗口
    plt.close()  # 关闭图形，释放内存

    # ==================== 步骤11: 返回分析结果 ====================
    # 【返回】字典格式，便于后续处理和统计
    return {
        'image_name': Path(image_path).name,  # 图像文件名
        'porosity': porosity,  # 孔隙率 (%)
        'longest_crack_length': max_length,  # 最长裂缝长度 (像素)
        'crack_pixels': crack_pixels,  # 裂缝总像素数
        'total_pixels': total_pixels,  # 图像总像素数
        'num_cracks': len(valid_components),  # 检测到的裂缝数量
        'output_path': str(output_path)  # 结果图像保存路径
    }

def main():
    """
    单张图片裂缝检测与分析
    """
    # ==================== 程序启动信息 ====================
    print("\n" + "="*60)
    print("OpenCV 裂缝检测与分析系统 (单张图片模式)")
    print("使用灰度检测、OTSU阈值和自适应阈值方法")
    print("="*60)

    # ==================== 设置单张图像路径 ====================
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    image_path = script_dir / "data" / "image (1).jpg"

    # ==================== 检查文件是否存在 ====================
    if not image_path.exists():
        print(f"\n错误: 找不到图像文件 '{image_path}'")
        print("请检查路径是否正确!")
        return

    # ==================== 处理单张图片 ====================
    print(f"\n正在处理: {image_path.name}")
    print("-" * 40)

    # 调用分析函数
    result = analyze_crack_with_opencv(image_path)

    if result:
        # ==================== 打印详细结果 ====================
        print(f"\n{'='*60}")
        print("分析结果:")
        print(f"{'='*60}")
        print(f"  图像名称: {result['image_name']}")
        print(f"  图像尺寸: {result['total_pixels']:,} 像素")
        print(f"  裂缝像素: {result['crack_pixels']:,}")
        print(f"  孔隙率: {result['porosity']:.3f}%")
        print(f"  最长裂缝: {result['longest_crack_length']:.0f} 像素")
        print(f"  裂缝数量: {result['num_cracks']}")
        print(f"  结果保存: {result['output_path']}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
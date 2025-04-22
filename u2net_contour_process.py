import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_mask_with_contours(mask_path, output_path=None, threshold=127, min_area=100):
    """
    处理U2Net生成的掩码图像，找到轮廓和旋转外接矩形，并标注宽高
    
    Args:
        mask_path: 掩码图像路径
        output_path: 输出图像路径，如果为None则不保存结果
        threshold: 二值化阈值，默认127
        min_area: 最小轮廓面积，小于此面积的轮廓将被忽略，默认100
        
    Returns:
        result_img: 带有标注的结果图像
        contours_info: 包含所有轮廓信息的列表，每个元素为字典，包含:
                      'contour': 轮廓点集
                      'rect': 旋转外接矩形 ((center_x, center_y), (width, height), angle)
                      'width': 宽度
                      'height': 高度
                      'area': 面积
    """
    # 读取掩码图像
    if isinstance(mask_path, str):
        mask = cv2.imread(mask_path)
    else:
        mask = mask_path.copy()
    
    # 转换为灰度图像
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 找到所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建彩色图像用于显示结果
    if len(mask.shape) == 2 or mask.shape[2] == 1:
        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        result_img = mask.copy()
    
    # 存储轮廓信息
    contours_info = []
    
    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 忽略小面积的轮廓
        if area < min_area:
            continue
        
        # 获取旋转外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # 获取宽度和高度
        width = rect[1][0]
        height = rect[1][1]
        
        # 确保宽度大于高度
        if width < height:
            width, height = height, width
        
        # 绘制轮廓
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        
        # 绘制旋转外接矩形
        cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
        
        # 获取矩形中心点
        center = (int(rect[0][0]), int(rect[0][1]))
        
        # 在图像上标注宽高
        cv2.putText(
            result_img, 
            f"H:{int(height)}", 
            (center[0] - 30, center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 255), 
            2
        )
        
        # 保存轮廓信息
        contours_info.append({
            'contour': contour,
            'rect': rect,
            'width': width,
            'height': height,
            'area': area
        })
    
    # 保存结果
    if output_path is not None:
        cv2.imwrite(output_path, result_img)
    
    return result_img, contours_info

def process_folder(input_dir, output_dir, threshold=127, min_area=100):
    """
    处理文件夹中的所有掩码图像
    
    Args:
        input_dir: 输入掩码图像目录
        output_dir: 输出结果图像目录
        threshold: 二值化阈值
        min_area: 最小轮廓面积
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有掩码图像
    mask_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(mask_files)} mask images in {input_dir}")
    
    # 处理每个掩码图像
    for mask_file in mask_files:
        mask_path = os.path.join(input_dir, mask_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(mask_file)[0]}_contours.png")
        
        print(f"Processing {mask_path}")
        
        try:
            # 处理掩码图像
            _, contours_info = process_mask_with_contours(
                mask_path, 
                output_path,
                threshold=threshold,
                min_area=min_area
            )
            
            print(f"Found {len(contours_info)} contours in {mask_file}")
            
        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
            continue
    
    print(f"Processing completed. Results saved in {output_dir}")

# 测试代码
if __name__ == "__main__":
    # 示例使用
    input_dir = "test_results/"  # U2Net推理结果目录
    output_dir = "contour_results/"  # 轮廓处理结果目录
    
    # 处理整个文件夹
    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=127,  # 二值化阈值
        min_area=100    # 最小轮廓面积
    ) 
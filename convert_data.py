#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将data目录中的jpg图像和对应的json标注文件中的polygon标注转换为掩码，
保存为与原图像相同文件名的PNG掩码图像
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# 创建目录结构
def create_dirs(output_dir="train"):
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    print(f"目录创建成功: {output_dir}/img, {output_dir}/masks")

# 处理单个图像和json对
def process_image(jpg_path, json_path, output_dir="train"):
    # 读取图像
    image = Image.open(jpg_path)
    width, height = image.size
    
    # 读取JSON标注
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建一个单一的掩码图像
    mask = Image.new('L', (width, height), 0)
    
    # 填充掩码，只处理polygon类型的标注
    polygon_count = 0
    for shape in data['shapes']:
        shape_type = shape.get('shape_type', '')
        
        if shape_type == 'polygon':
            draw = ImageDraw.Draw(mask)
            points = [tuple(point) for point in shape['points']]
            draw.polygon(points, fill=255)
            polygon_count += 1
    
    # 保存原始图像到img目录
    image_filename = os.path.basename(jpg_path)
    image_out_path = os.path.join(output_dir, "img", image_filename)
    image.save(image_out_path)
    
    # 保存掩码图像到masks目录，使用原图像的文件名（不带标签后缀）
    base_filename = os.path.splitext(os.path.basename(jpg_path))[0]
    mask_filename = f"{base_filename}.png"
    mask_out_path = os.path.join(output_dir, "masks", mask_filename)
    mask.save(mask_out_path)
    
    return polygon_count

def main():
    # 设置输入和输出目录
    data_dir = "data"
    output_dir = "train"
    
    # 创建目录
    create_dirs(output_dir)
    
    # 获取data目录中的所有文件
    files = os.listdir(data_dir)
    
    # 找出所有的图像文件
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"警告: 在 {data_dir} 目录中未找到图像文件")
        return
    
    total_polygons = 0
    processed_files = 0
    
    for image_file in tqdm(image_files, desc="处理图像"):
        # 构建对应的json文件名
        base_name = os.path.splitext(image_file)[0]
        json_file = f"{base_name}.json"
        
        # 检查json文件是否存在
        if json_file in files:
            image_path = os.path.join(data_dir, image_file)
            json_path = os.path.join(data_dir, json_file)
            
            # 处理图像和json对
            try:
                num_polygons = process_image(image_path, json_path, output_dir)
                total_polygons += num_polygons
                processed_files += 1
                print(f"已处理: {image_file}，包含 {num_polygons} 个多边形标注")
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
        else:
            print(f"警告: 找不到 {image_file} 对应的json文件")
    
    print(f"完成! 已处理 {processed_files} 个图像文件，共处理 {total_polygons} 个多边形标注")

if __name__ == "__main__":
    main() 
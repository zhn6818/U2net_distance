# 图像分割数据处理工具

这个项目提供了一个简单工具，用于将JSON标注文件中的多边形(polygon)标注转换为二值掩码图像，用于图像分割模型训练，以及使用U2Net模型进行图像分割和分割结果后处理。

## 功能特点

- 将图像文件和对应的JSON标注文件转换为训练图像和分割掩码
- 只处理polygon类型的标注，生成对应的二值掩码
- 将所有polygon标注合并到单个掩码中
- 自动将转换后的数据保存到train/img和train/masks目录中
- 使用U2Net模型对图像进行分割预测
- **新增：对分割结果进行轮廓分析，检测旋转外接矩形并标注宽高**

## 文件结构

```
- convert_data.py             # 数据转换脚本
- u2net_train.py              # 模型训练脚本
- u2net_infer_single_img.py   # 单图像/文件夹推理脚本
- u2net_contour_process.py    # 轮廓处理脚本
- requirements.txt            # 依赖项列表
- data/                       # 原始数据目录
  - *.jpg/png                 # 原始图像文件
  - *.json                    # 对应的JSON标注文件
- train/                      # 处理后的训练数据（脚本自动创建）
  - img/                      # 图像文件夹
  - masks/                    # 掩码文件夹，包含与原图像同名的掩码
- test_results/               # 推理结果输出目录
- contour_results/            # 轮廓处理结果输出目录
```

## 使用方法

### 1. 安装依赖项

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将要处理的图片（JPG/PNG/BMP格式）和对应的JSON标注文件放置在`data`目录下。每个图片应该有一个同名的JSON文件。

### 3. 数据准备

```bash
python convert_data.py
```

### 4. 模型训练

```bash
python u2net_train.py
```

### 5. 图像分割与轮廓分析

```bash
python u2net_infer_single_img.py
```

默认情况下，脚本会:
1. 从`train/img/`目录读取图像
2. 使用U2Net模型进行推理
3. 将分割掩码保存到`test_results/`目录
4. 对分割掩码进行轮廓分析，检测旋转外接矩形并标注宽高
5. 将轮廓分析结果保存到`contour_results/`目录

## JSON标注格式

脚本支持处理以下格式的JSON标注文件:

```json
{
  "version": "x.x.x",
  "shapes": [
    {
      "label": "类别名称",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon",
      ...
    },
    ...
  ],
  ...
}
```

## 轮廓分析功能说明

新增的轮廓分析功能可以:
1. 检测分割掩码中的所有轮廓
2. 计算每个轮廓的旋转外接矩形
3. 在结果图像上标注矩形的宽度和高度
4. 过滤掉小于指定面积的轮廓

可以通过以下参数控制轮廓分析功能:
- `process_contours`: 是否启用轮廓分析（默认：True）
- `contour_threshold`: 二值化阈值（默认：127）
- `min_contour_area`: 最小轮廓面积（默认：100）

## 注意事项

- 脚本只处理JSON中的polygon类型标注
- 生成的掩码是二值图像，255表示前景，0表示背景
- 图像文件支持.jpg, .jpeg, .png和.bmp格式
- 所有标签的polygon都会合并到同一个掩码文件中
- 轮廓分析功能需要安装OpenCV库 
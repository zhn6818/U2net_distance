# 图像分割数据处理工具

这个项目提供了一个简单工具，用于将JSON标注文件中的多边形(polygon)标注转换为二值掩码图像，用于图像分割模型训练。

## 功能特点

- 将图像文件和对应的JSON标注文件转换为训练图像和分割掩码
- 只处理polygon类型的标注，生成对应的二值掩码
- 将所有polygon标注合并到单个掩码中
- 自动将转换后的数据保存到train/img和train/masks目录中

## 文件结构

```
- convert_data.py   # 主脚本文件
- requirements.txt  # 依赖项列表
- data/             # 原始数据目录
  - *.jpg/png       # 原始图像文件
  - *.json          # 对应的JSON标注文件
- train/            # 处理后的训练数据（脚本自动创建）
  - img/            # 图像文件夹
  - masks/          # 掩码文件夹，包含与原图像同名的掩码
```

## 使用方法

### 1. 安装依赖项

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将要处理的图片（JPG/PNG/BMP格式）和对应的JSON标注文件放置在`data`目录下。每个图片应该有一个同名的JSON文件。

### 3. 运行脚本

```bash
python convert_data.py
```

### 4. 输出结果

脚本处理完成后，会在输出目录下生成:
- `img/`: 包含原始图像的副本
- `masks/`: 包含生成的掩码图像
  - 每个掩码文件与原图像同名，但扩展名为.png

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

## 注意事项

- 脚本只处理JSON中的polygon类型标注
- 生成的掩码是二值图像，255表示前景，0表示背景
- 图像文件支持.jpg, .jpeg, .png和.bmp格式
- 所有标签的polygon都会合并到同一个掩码文件中 
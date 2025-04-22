import os
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import U2NET
from model import U2NETP
from data_loader import RescaleT
from data_loader import ToTensorLab

# 导入轮廓处理模块
from u2net_contour_process import process_mask_with_contours

pred_size = 1024

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def get_device():
    """获取可用的设备类型：CUDA、MPS 或 CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def read_image(image_path):
    """读取图片并进行基础处理"""
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    return image

def inference_folder(image_dir, model_path, output_dir, contour_dir=None, model_type='u2net', 
                    process_contours=False, contour_threshold=127, min_contour_area=100):
    """
    对文件夹中的所有图片进行推理
    Args:
        image_dir: 输入图片文件夹路径
        model_path: 模型权重文件路径
        output_dir: 掩码输出目录
        contour_dir: 轮廓处理结果输出目录，如果为None则与output_dir相同
        model_type: 使用的模型类型，'u2net' 或 'u2netp'
        process_contours: 是否进行轮廓处理
        contour_threshold: 轮廓检测的二值化阈值
        min_contour_area: 最小轮廓面积
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif model_type == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # 图像预处理
    transform = transforms.Compose([
        RescaleT(pred_size),
        ToTensorLab(flag=0)
    ])

    # 获取所有图片文件
    img_name_list = glob.glob(os.path.join(image_dir, '*.*'))
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_name_list = [f for f in img_name_list if os.path.splitext(f)[1].lower() in supported_formats]
    
    print(f"Found {len(img_name_list)} images in {image_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要轮廓处理，确保轮廓输出目录存在
    if process_contours:
        if contour_dir is None:
            contour_dir = os.path.join(output_dir, 'contours')
        os.makedirs(contour_dir, exist_ok=True)
        print(f"Contour processing enabled. Results will be saved to {contour_dir}")
    
    # 处理每张图片
    for image_path in img_name_list:
        print(f"Processing {image_path}")
        try:
            # 读取和处理图像
            image = read_image(image_path)
            label = np.zeros(image.shape[0:2])
            label = label[:,:,np.newaxis]  # Add channel dimension
            
            # 准备输入数据
            sample = {'imidx': np.array([0]), 'image': image, 'label': label}
            sample = transform(sample)
            inputs_test = sample['image']
            inputs_test = inputs_test.unsqueeze(0)
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = Variable(inputs_test).to(device)

            # 推理
            with torch.no_grad():
                d1,d2,d3,d4,d5,d6,d7 = net(inputs_test)
                pred = d1[:,0,:,:]
                pred = normPRED(pred)

                # 处理预测结果
                predict = pred.squeeze()
                predict_np = predict.cpu().data.numpy()
                im = Image.fromarray(predict_np*255).convert('RGB')
                
                # 调整回原始图像大小
                image = io.imread(image_path)
                imo = im.resize((image.shape[1],image.shape[0]), resample=Image.BILINEAR)

                # 保存掩码结果
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(output_dir, f"{img_name}.png")
                imo.save(mask_path)
                
                # 如果启用轮廓处理，处理掩码图像
                if process_contours:
                    contour_output_path = os.path.join(contour_dir, f"{img_name}_contours.png")
                    
                    # 将PIL图像转换为numpy数组用于轮廓处理
                    mask_np = np.array(imo)
                    
                    # 处理掩码图像，查找轮廓和外接矩形
                    _, contours_info = process_mask_with_contours(
                        mask_np,
                        contour_output_path,
                        threshold=contour_threshold,
                        min_area=min_contour_area
                    )
                    
                    print(f"Found {len(contours_info)} contours in {img_name}")

                del d1,d2,d3,d4,d5,d6,d7

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"Processing completed. Results saved in {output_dir}")
    if process_contours:
        print(f"Contour processing results saved in {contour_dir}")

if __name__ == "__main__":
    # 示例使用
    model_type = 'u2net'
    image_dir = 'train/img/'
    model_path = 'saved_models/u2net/u2net_best_acc_0.9995_epoch_78.pth'
    output_dir = 'test_results/'
    
    # 启用轮廓处理
    process_contours = True
    contour_dir = 'contour_results/'
    
    inference_folder(
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir,
        contour_dir=contour_dir,
        model_type=model_type,
        process_contours=process_contours,
        contour_threshold=127,  # 二值化阈值
        min_contour_area=100    # 最小轮廓面积
    ) 
import numpy as np
import torch

from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


# Dice evaluation
def dice_coefficient(tensor1, tensor2):
    """
    计算两张 PyTorch 张量图像的 Dice 系数

    Parameters:
    - tensor1: torch.Tensor, 第一张图像的张量
    - tensor2: torch.Tensor, 第二张图像的张量

    Returns:
    - dice_score: float, Dice 系数
    """
    # 将张量转换为 NumPy 数组
    array1 = tensor1
    array2 = tensor2

    # 将数组转换为一维数组
    flat_array1 = array1.flatten()
    flat_array2 = array2.flatten()

    # 计算交集和并集的大小
    intersection = (flat_array1 * flat_array2).sum()
    union = flat_array1.sum() + flat_array2.sum()

    # 计算 Dice 系数
    dice_score = (2.0 * intersection) / union

    return dice_score


# HD95
def hd95(image1, image2):
    # 计算两张图像的差异
    differences = torch.abs(image1 - image2)

    # 将差异的像素值展平为一维数组
    differences_flat = differences.view(-1)

    # 计算像素值的95%分位数
    hd95_value = torch.kthvalue(differences_flat, int(0.95 * differences_flat.numel()))[0].item()

    return hd95_value


# ASD
def compute_surface_distance(image1, image2):
    # Ensure that the input images are in the range [0, 1]
    assert torch.all((image1 >= 0) & (image1 <= 1)), "Image1 should be in the range [0, 1]"
    assert torch.all((image2 >= 0) & (image2 <= 1)), "Image2 should be in the range [0, 1]"

    # Convert to binary masks
    threshold = 0.5
    binary_mask1 = (image1 > threshold).float()
    binary_mask2 = (image2 > threshold).float()

    # Compute surface distance for each channel
    asd_channels = [compute_surface_distance(binary_mask1[channel], binary_mask2[channel]) for channel in range(3)]

    # Calculate average ASD across channels
    asd_average = sum(asd_channels) / 3

    return asd_average


# ASSD
def assd(image1, image2):
    asd_value = compute_surface_distance(image1, image2)
    assd_value = 0.5 * (asd_value + compute_surface_distance(image2, image1))
    return assd_value

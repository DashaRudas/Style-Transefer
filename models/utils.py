
# utils.py

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

def calc_mean_std(feat, eps=1e-5):
    """
    Вычисляем среднее и стандартное отклонение по каждому каналу
    """
    N, C = feat.shape[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat, eps=1e-5):
    """
    Adaptive Instance Normalization
    Нормируем контент-фичи по mean/std, масштабируем и сдвигает их mean/std от стиля.
    """
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

def denormalize(tensor):
    """
    Обратная нормализация
    """
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)  # сделать (1,3,H,W) для удобства
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def load_image(path, transform=None):
    img = Image.open(path).convert('RGB')
    if transform:
        img = transform(img)
    return img

def save_image(tensor, path=None):
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    pil_image = T.ToPILImage()(tensor.cpu())
    if path:
        pil_image.save(path)
    return pil_image




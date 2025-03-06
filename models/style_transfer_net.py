import torch
import torch.nn as nn

from .utils import adain
from .utils import calc_mean_std

class StyleTransferNet(nn.Module):
    """
    Объединям encoder и decoder, по сути применение AdaIN
    """
    def __init__(self, encoder, decoder):
        super(StyleTransferNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, content, style, alpha=1.0)
        content_feat = self.encoder(content)
        style_feat   = self.encoder(style)
        # AdaIN для получения стилизованных фичей
        t = adain(content_feat, style_feat)
        # Смешиваем стилизованные фичи с исходными (если alpha < 1.0)
        t = alpha * t + (1 - alpha) * content_feat
        out = self.decoder(t)
        return out



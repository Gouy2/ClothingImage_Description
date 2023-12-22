import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ViTEncoder, self).__init__()
        # 加载预训练的 Vision Transformer 模型
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # finetuned=False 不微调，冻结权重
        if not finetuned:
            for param in self.model.parameters():
                param.requires_grad = False
        # else:
            # 只冻结前几层的参数
            # for param in list(self.model.parameters())[:num_frozen_layers]:
            #     param.requires_grad = False


        self.model.head = nn.Identity()  # 空层，输出特征表示


    def forward(self, images):
        # 通过 Vision Transformer 提取特征
        features = self.model(images)
        return features
    
    

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self, finetuned=True, output_dim=512):
        super(ViTEncoder, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        if not finetuned:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.head = nn.Identity()  # 移除原始的头部分类层

        # 添加一个新的线性层以匹配 word_dim
        self.adapt_layer = nn.Linear(1000, output_dim)  # 假设 ViT 输出为 1000 维

    def forward(self, images):
        features = self.model(images)
        # 调整特征维度
        adapted_features = self.adapt_layer(features)
        # 重塑为 [batch_size, 1, output_dim]
        # adapted_features = adapted_features.unsqueeze(1)

        # print("ViTEncoder output shape:", adapted_features.shape)
        return adapted_features

    
    

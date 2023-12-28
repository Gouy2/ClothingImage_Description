import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet101_Weights


#修改了resnet编码器
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # 保留ResNet-101的所有层，除了最后两层
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-2]))
        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.feature_extractor.parameters():
            param.requires_grad = finetuned

    def forward(self, images):
        # 通过特征提取层
        out = self.feature_extractor(images)
        # 应用全局平均池化，输出的大小为 (batch_size, channels, 1, 1)
        out = self.global_avg_pool(out)
        # 展平输出，使其形状变为 (batch_size, channels)
        out = out.view(out.size(0), -1)
        return out

#修改了解码器中的隐藏层初始化
def init_hidden_state(self, image_code, captions, cap_lens):
    """
    参数：
        image_code：图像编码器输出的图像表示 
                    (batch_size, image_code_dim)
    """
    batch_size, image_code_dim = image_code.size(0), image_code.size(1)
    
    # （1）按照caption的长短排序
    sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
    captions = captions[sorted_cap_indices]
    image_code = image_code[sorted_cap_indices]

    #（2）初始化隐状态
    # 注意：由于image_code已经是整体表示，不再需要对其进行平均
    hidden_state = self.init_state(image_code)
    hidden_state = hidden_state.view(
                        batch_size, 
                        self.rnn.num_layers, 
                        self.rnn.hidden_size).permute(1, 0, 2)
    return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state


#修改部分束搜索
#image_code = image_code.unsqueeze(0).repeat(beam_k, 1)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# 交叉熵损失函数
class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        """
        参数：
            predictions：按文本长度排序过的预测结果
            targets：按文本长度排序过的文本描述
            lengths：文本长度
        """
        predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        return self.loss_fn(predictions, targets)
    

class TransformerCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TransformerCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        """
        参数：
            predictions：预测结果
            targets：文本描述
            lengths：文本长度
        """
        # 使用 enforce_sorted=False 允许非降序排列的序列
        predictions_packed = pack_padded_sequence(predictions, lengths, batch_first=True, enforce_sorted=False)[0]
        targets_packed = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]
        return self.loss_fn(predictions_packed, targets_packed)
    

        

# 优化器
def get_optimizer(model, config):
    return torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, model.encoder.parameters()), 
                              "lr": config.encoder_learning_rate},
                             {"params": filter(lambda p: p.requires_grad, model.decoder.parameters()), 
                              "lr": config.decoder_learning_rate}])

# 调整学习速率
def adjust_learning_rate(optimizer, epoch, config):
    if epoch < config.warmup_epochs:
        # 预热阶段，学习率线性增加
        encoder_lr = config.encoder_learning_rate * (epoch / config.warmup_epochs)
        decoder_lr = config.decoder_learning_rate * (epoch /config.warmup_epochs)
    else:
        # 预热后，学习率线性减少
        encoder_lr = config.encoder_learning_rate * (1 - (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs))
        decoder_lr = config.decoder_learning_rate * (1 - (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs))

    # 调整编码器和解码器的学习率
    optimizer.param_groups[0]['lr'] = encoder_lr
    optimizer.param_groups[1]['lr'] = decoder_lr

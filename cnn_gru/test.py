import torch
from torchvision import transforms
from PIL import Image
import json

from arctic import ARCTIC

vocab_path = '../data/cloth/vocab.json'

checkpoint = './model/best_model.ckpt' # 验证集上表现最优的模型的路径
# checkpoint = './model/last_model.ckpt', # 训练完成时的模型的路径


# 加载模型
checkpoint = torch.load(checkpoint)
model = checkpoint['model']

model.eval()  # 将模型设置为评估模式

# 加载词汇表
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加批次维度

# 解码模型输出
def decode_predictions(output, vocab):
    # 将索引转换为文字
    words = [vocab.get(idx.item(), '<unk>') for idx in output]
    return ' '.join(words)

# 推理
def generate_description(image_path, model, vocab):
    image = preprocess_image(image_path)
    with torch.no_grad():  # 不需要计算梯度
        output = model(image)
        # 以下是一个示例，取决于您的模型输出格式
        predicted_indices = output.argmax(dim=2)
        description = decode_predictions(predicted_indices[0], vocab)
    return description

# 使用模型
image_path = '../data/cloth/images/WOMEN-Tees_Tanks-id_00003687-13_4_full.jpg'
description = generate_description(image_path, model, vocab)
print(description)

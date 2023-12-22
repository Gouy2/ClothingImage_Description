import torch
import sys
import json
from PIL import Image
import torchvision.transforms as transforms


def generate_caption(image_path, trained_model, transform):
    """
    生成图片的文字描述。

    参数:
    image_path: 输入图片的路径
    model: 训练好的模型
    transform: 图像预处理的流程
    vocab: 词汇表
    max_length: 生成描述的最大长度

    返回:
    生成的文字描述
    """
    # 图像加载和预处理
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(trained_model, map_location=device)
    model = checkpoint['model']

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 生成描述
    with torch.no_grad():
        caption = model.generate_by_beamsearch(image, beam_k=5, max_len=120)

    return caption


def indices_to_sentence_nested(indices_list, vocab):
    """
    将嵌套单词索引列表转换为对应的单词句子，并去除特殊标记。

    参数:
    indices_list: 嵌套单词索引的列表
    vocab: 词汇表，索引到单词的映射

    返回:
    单词组成的句子字符串
    """
    # 词汇表反向映射：从索引找到单词
    reverse_vocab = {int(v): k for k, v in vocab.items()}

    # 处理可能的嵌套列表
    if indices_list and isinstance(indices_list[0], list):
        indices = indices_list[0]
    else:
        indices = indices_list

    # 转换索引到单词，并去除特殊标记
    words = [reverse_vocab.get(index, '') for index in indices if reverse_vocab.get(index, '') not in ['<start>', '<end>', '<pad>', '<unk>']]

    # 将单词列表拼接成句子
    sentence = ' '.join(words)

    return sentence


# 使用示例
# if __name__ == '__main__':
    # caption = generate_caption('../data/cloth/test.jpg', model, transform)
    # caption_words = indices_to_sentence_nested(caption, vocab)
    # print("Generated Caption:", caption_words)

    # app = QApplication(sys.argv)
    # ex = ImageCaptioningApp(model,vocab,transform)
    # ex.show()
    # sys.exit(app.exec_())

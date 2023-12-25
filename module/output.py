import torch

import json
from PIL import Image
import torchvision.transforms as transforms
import re
from translate import Translator
import sys
sys.path.append("../gru") #model定义路径

def generate_caption(image_path, trained_model):
    """
    生成图片的文字描述。

    参数:
    image_path: 输入图片的路径
    trained_model: 训练好的模型


    返回:
    生成的文字描述
    """
    # 图像加载和预处理
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(trained_model, map_location=device)
    model = checkpoint['model']

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 生成描述
    with torch.no_grad():
        caption = model.generate_by_beamsearch(image, beam_k=5, max_len=120)

        # if caption is not str


    return caption


def indices_to_sentence_nested(indices_list, vocab_path):
    """
    将嵌套单词索引列表转换为对应的单词句子，并去除特殊标记。

    参数:
    indices_list: 嵌套单词索引的列表
    vocab: 词汇表，索引到单词的映射

    返回:
    单词组成的句子字符串
    """
    # 词汇表反向映射：从索引找到单词
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

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

    # 去除多余的空格和标点
    sentence = re.sub(r' ,', ',', sentence)
    sentence = re.sub(r' \.', '.', sentence)

    # sentence = translate_caption(sentence) # 英语翻译中文

    # print(sentence)

    return sentence

def translate_caption(caption):
    # 英语翻译中文
        translator = Translator(from_lang="English",to_lang="chinese")
        # translation = translator.translate(caption)
        translation = translator.translate("The upper clothing has short sleeves, cotton fabric and pure color patterns.")
        
        print(translation)
        return translation


# 使用示例
if __name__ == '__main__':
    vocab = '../data/cloth/vocab.json'
    model = './model/g1-1.ckpt'
    # model = '.../save/gru/g1-1.ckpt'
    caption = generate_caption('../data/test.jpg', model)
    caption_words = indices_to_sentence_nested(caption, vocab)

    # test_translate = translate_caption(caption_words)

    # caption_words = str(caption_words)
    # print("Generated Caption:", caption_words)

    translator = Translator(from_lang="English",to_lang="chinese")
    # translation = translator.translate(caption_words)
    translation = translator.translate("The upper clothing has short sleeves,cotton fabric and pure color patterns.")
        
    print(translation)

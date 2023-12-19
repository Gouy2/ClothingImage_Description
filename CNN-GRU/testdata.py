import os
import json
import random
import re

from collections import Counter
import itertools


captions_per_image = 5

file_path = "../data/cloth"
image_path = "../data/cloth/images"
vocab_path = "../data/cloth/vocab.json"


# with open(file_path, 'r') as file:
#     original_data = json.load(file)

# # 初始化转换后的结构
# transformed_data = {"IMAGES": [], "CAPTIONS": []}

# # 填充转换后的结构
# for image_name, captions in original_data.items():
#     # 添加图片名称
#     transformed_data["IMAGES"].append(image_name)
    
#     # 确保描述是字符串格式，如果不是，转换为字符串
#     if not isinstance(captions, str):
#         captions = ' '.join(captions)


#     split_captions = [caption.strip() for caption in captions.split('.') if caption.strip()]


#     # print(split_captions)
#     # print('-------------------')

#     for i in range(len(split_captions)):
#         # print(split_captions[i])
#         if len(split_captions[i]) < captions_per_image:
#             split_captions = split_captions[i] + \
#                 [random.choice(split_captions[i]) for _ in range(captions_per_image - len(split_captions[i]))]
    

#     # 添加描述
#     for caption in split_captions:
#         transformed_data["CAPTIONS"].append([caption])


# # 合并所有描述为一个长文本
# all_captions = [item for sublist in transformed_data["CAPTIONS"] for item in sublist]
# all_text = " ".join(all_captions)

# # 分割文本为单词并统计频率
# words = all_text.split()
# word_counts = Counter(words)

# # 创建词典并排序（最常见的单词排在前面）
# vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# # 添加特殊标识符
# special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
# vocab = special_tokens + vocab

# # 将词典转换为 {word: index} 形式
# word_to_index = {word: index for index, word in enumerate(vocab)}

# # 展示词典的前几个条目以确认正确
# print(list(itertools.islice(word_to_index.items(), 10)))

# with open(vocab_path, 'w') as fw:
#         json.dump(word_to_index, fw)



# 定义一个函数来清理文本，移除逗号
def clean_text(text):
    # 移除逗号并返回清理后的文本
    return re.sub(r',', '', text)

def process_punctuation(text):
    # 在逗号和句号前添加空格
    processed_text = re.sub(r',', ' ,', text)
    processed_text = re.sub(r'\.', ' .', processed_text)
    return processed_text

with open(os.path.join(file_path, 'test_captions.json'), 'r') as file1:
    test_captions_data = json.load(file1)

with open(os.path.join(file_path, 'train_captions.json'), 'r') as file2:
    train_captions_data = json.load(file2)

# 提取 .json 中的所有描述
all_test_captions = [caption for image, caption in test_captions_data.items()]
all_train_captions =[caption for image, caption in train_captions_data.items()] 

# 合并两个文件中的描述，并分割为单词
all_captions_combined = all_test_captions + all_train_captions
processed_captions_combined = [process_punctuation(caption) for caption in all_captions_combined]
all_words_combined = " ".join(processed_captions_combined).split()
# print(all_words_combined)

# 统计词频
word_counts_combined = Counter(all_words_combined)

# 创建词典并排序
vocab_combined = sorted(word_counts_combined, key=word_counts_combined.get, reverse=True)

# 添加特殊标识符
special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
vocab_combined = special_tokens + vocab_combined

# 转换为 {word: index} 形式的词典
word_to_index_combined = {word: index for index, word in enumerate(vocab_combined)}

# 展示新词典的前几个条目
print(list(itertools.islice(word_to_index_combined.items(), 10)))

with open(vocab_path, 'w') as fw:
        json.dump(word_to_index_combined, fw)

# 展示转换后结构的前几个条目以确认转换正确
# sample_transformed_data = {
#     "IMAGES": transformed_data["IMAGES"][:2],
#     "CAPTIONS": transformed_data["CAPTIONS"][:10]  # 展示前10个描述
# }

# print(sample_transformed_data)
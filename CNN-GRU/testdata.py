import os
import json
import random
import re

from collections import Counter
import itertools


target_num_captions = 5

file_path = "../data/cloth"
image_path = "../data/cloth/images"
vocab_path = "../data/cloth/vocab.json"


with open(os.path.join(file_path, 'test_captions.json'), 'r') as file1:
    test_captions_data = json.load(file1)

with open(os.path.join(file_path, 'train_captions.json'), 'r') as file2:
    train_captions_data = json.load(file2)

# 初始化转换后的结构
transformed_data = {"IMAGES": [], "CAPTIONS": []}


for image_name, captions in test_captions_data.items():
    transformed_data["IMAGES"].append(os.path.join(file_path, image_name))

    # 确保描述是列表格式
    if not isinstance(captions, list):
        captions = [captions]

    # if not isinstance(captions, str):
    #     captions = ' '.join(captions)


    # 分割并处理每个描述
    for caption in captions:
        split_captions = [caption.strip() for caption in caption.split('.') if caption.strip()]
        

        if len(split_captions) < target_num_captions:
        # 复制描述
            split_captions += [random.choice(split_captions) for _ in range(target_num_captions - len(split_captions))]
        elif len(split_captions) > target_num_captions:
        # 随机选择描述
            split_captions = random.sample(split_captions, target_num_captions)

        # print (split_captions)
        # print("**************")

        for split_caption in split_captions:
            transformed_data["CAPTIONS"].append([split_caption])



def process_punctuation(text):
    # 在逗号和句号前添加空格
    processed_text = re.sub(r',', ' ,', text)
    processed_text = re.sub(r'\.', ' .', processed_text)
    return processed_text


#提取 .json 中的所有描述
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
word_to_index_combined = {word: index+1 for index, word in enumerate(vocab_combined)}

# print(list(itertools.islice(word_to_index_combined.items(), 10)))

with open(vocab_path, 'w') as fw:
        json.dump(word_to_index_combined, fw)


unk_index = word_to_index_combined["<unk>"]
transformed_data_with_indices = {"IMAGES": transformed_data["IMAGES"], "CAPTIONS": []}

for caption_list in transformed_data["CAPTIONS"]:
    # 检查 caption_list 是否为非空列表
    if caption_list:
        caption = caption_list[0]
        # 分割描述为单词
        words = caption.split()
        # 转换单词为索引
        caption_indices = [word_to_index_combined.get(word, unk_index) for word in words]
        # 添加转换后的描述到新结构中
        transformed_data_with_indices["CAPTIONS"].append(caption_indices)

with open(os.path.join(file_path, 'test_data.json'), 'w') as fw:
            json.dump(transformed_data_with_indices, fw)

# 用于处理 train_captions_data 的代码

# 初始化 train_data 的转换结构
transformed_train_data = {"IMAGES": [], "CAPTIONS": []}

for image_name, captions in train_captions_data.items():
    transformed_train_data["IMAGES"].append(os.path.join(file_path, image_name))

    # 确保描述是列表格式
    if not isinstance(captions, list):
        captions = [captions]

    for caption in captions:
        split_captions = [caption.strip() for caption in caption.split('.') if caption.strip()]
        
        if len(split_captions) < target_num_captions:
            split_captions += [random.choice(split_captions) for _ in range(target_num_captions - len(split_captions))]
        elif len(split_captions) > target_num_captions:
            split_captions = random.sample(split_captions, target_num_captions)

        for split_caption in split_captions:
            transformed_train_data["CAPTIONS"].append([split_caption])

# 转换 train_data 中的描述为索引
transformed_train_data_with_indices = {"IMAGES": transformed_train_data["IMAGES"], "CAPTIONS": []}

for caption_list in transformed_train_data["CAPTIONS"]:
    if caption_list:
        caption = caption_list[0]
        words = caption.split()
        caption_indices = [word_to_index_combined.get(word, unk_index) for word in words]
        transformed_train_data_with_indices["CAPTIONS"].append(caption_indices)

# 保存处理后的 train_data
with open(os.path.join(file_path, 'train_data.json'), 'w') as fw:
    json.dump(transformed_train_data_with_indices, fw)


# #展示转换后结构的前几个条目以确认转换正确
# sample_transformed_data = {
#     "IMAGES": transformed_data_with_indices["IMAGES"][:2],
#     "CAPTIONS": transformed_data_with_indices["CAPTIONS"][:10]  # 展示前10个描述
# }

# print(sample_transformed_data)
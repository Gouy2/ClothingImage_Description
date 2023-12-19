import os
import json
import random
import re

from collections import Counter
import itertools

file_path = "../data/cloth"
vocab_path = "../data/cloth/vocab.json"


def process_captions(dataset_type,
                     
                    image_path = "../data/cloth/images",
                    
                    target_num_captions = 5):
    """
    处理每张图片的描述，确保每张图片对应五个描述。
    """

    file_name = f'{dataset_type}_captions.json'
    print(file_name)
    with open(os.path.join(file_path, file_name), 'r') as file:
        captions_data = json.load(file)

    transformed_data = {"IMAGES": [], "CAPTIONS": []}
    for image_name, captions in captions_data.items():
        transformed_data["IMAGES"].append(os.path.join(image_path, image_name))
        
        if not isinstance(captions, list):
            captions = [captions]

        for caption in captions:
            split_captions = [caption.strip() for caption in caption.split('.') if caption.strip()]

            if len(split_captions) < target_num_captions:
                split_captions += [random.choice(split_captions) for _ in range(target_num_captions - len(split_captions))]
            elif len(split_captions) > target_num_captions:
                split_captions = random.sample(split_captions, target_num_captions)

            for split_caption in split_captions:
                transformed_data["CAPTIONS"].append([split_caption])
    
    return transformed_data

def process_punctuation(text):
    # 在逗号和句号前添加空格
    processed_text = re.sub(r',', ' ,', text)
    processed_text = re.sub(r'\.', ' .', processed_text)
    return processed_text

def process_vocab(test_captions_data, train_captions_data, vocab_path):
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

    return word_to_index_combined, unk_index

def convert_captions_to_indices(transformed_data, word_to_index, unk_index):
    """
    将描述中的单词转换为对应的词典索引。
    """
    transformed_data_with_indices = {"IMAGES": transformed_data["IMAGES"], "CAPTIONS": []}
    for caption_list in transformed_data["CAPTIONS"]:
        if caption_list:
            caption = caption_list[0]
            words = caption.split()
            caption_indices = [word_to_index.get(word, unk_index) for word in words]
            transformed_data_with_indices["CAPTIONS"].append(caption_indices)
    return transformed_data_with_indices

def save_data_to_json(data, file_path):
    """
    保存处理后的数据到 JSON 文件。
    """
    with open(file_path, 'w') as fw:
        json.dump(data, fw)

# 处理 test_captions_data 和 train_captions_data
transformed_test_data = process_captions("test")
transformed_train_data = process_captions("train")

# 处理词典
word_to_index_combined, unk_index = process_vocab(transformed_test_data, transformed_train_data, os.path.join(file_path, 'vocab.json'))



# 转换为索引
transformed_test_data_with_indices = convert_captions_to_indices(transformed_test_data, word_to_index_combined, unk_index)
transformed_train_data_with_indices = convert_captions_to_indices(transformed_train_data, word_to_index_combined, unk_index)

# 保存为 JSON
save_data_to_json(transformed_test_data_with_indices, os.path.join(file_path, 'test_data.json'))
save_data_to_json(transformed_train_data_with_indices, os.path.join(file_path, 'train_data.json'))


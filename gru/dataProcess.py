import os
import json
import random
import re

from collections import Counter

file_path = "../data/cloth"
vocab_path = "../data/cloth/vocab.json"
image_path = "../data/cloth/images"

def create_dataset(file_path,vocab_path,image_path,target_num_captions = 5):
    def process_captions(dataset_type,
                        ):
        """
        处理每张图片的描述，确保每张图片对应五个描述。
        """

        file_name = f'{dataset_type}_captions.json'
        # print(file_name)
        # print("载入数据集:",file_name)
        with open(os.path.join(file_path, file_name), 'r') as file:
            captions_data = json.load(file)


        transformed_data = {"IMAGES": [], "CAPTIONS": []}
        for image_name, captions in captions_data.items():
            transformed_data["IMAGES"].append(os.path.join(image_path, image_name))
            
            captions = process_punctuation(captions)

            if not isinstance(captions, list):
                captions = [captions]


             
            transformed_data["CAPTIONS"].append(captions)

            # for caption in captions:
            #     split_captions = [caption.strip() for caption in caption.split('.') if caption.strip()]
                


            #     if len(split_captions) < target_num_captions:
            #         split_captions += [random.choice(split_captions) for _ in range(target_num_captions - len(split_captions))]
            #     elif len(split_captions) > target_num_captions:
            #         split_captions = random.sample(split_captions, target_num_captions)

            #     for split_caption in split_captions:
            #         transformed_data["CAPTIONS"].append([split_caption])
        
        # print(transformed_data["CAPTIONS"][0])

        return transformed_data

    def process_punctuation(text):
        # 在逗号和句号前添加空格
        processed_text = re.sub(r',', ' ,', text)
        processed_text = re.sub(r'\.', ' .', processed_text)
        return processed_text

    def process_vocab(test_captions_data, train_captions_data, vocab_path):
        # 提取 .json 中的所有描述
        all_test_captions = []
        all_train_captions = []

        for image, captions in test_captions_data.items():
            if isinstance(captions, list):
                all_test_captions.extend(captions)
            else:
                all_test_captions.append(captions)

        for image, captions in train_captions_data.items():
            if isinstance(captions, list):
                all_train_captions.extend(captions)
            else:
                all_train_captions.append(captions)

        # 确保所有描述都是字符串
        all_captions_combined = []
        for caption in all_test_captions + all_train_captions:
            if isinstance(caption, str):
                all_captions_combined.append(caption)
            elif isinstance(caption, list):
                all_captions_combined.extend(caption)

        # 处理标点符号
        processed_captions_combined = [process_punctuation(caption) for caption in all_captions_combined if isinstance(caption, str)]

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
                
                caption_indices =  [word_to_index['<start>']] + [word_to_index.get(word, unk_index) for word in words]+ [word_to_index['<end>']]
                # caption_indices =  [word_to_index.get(word, unk_index) for word in words]
                transformed_data_with_indices["CAPTIONS"].append(caption_indices)

        # print(len(transformed_data_with_indices["IMAGES"]))
        # print(len(transformed_data_with_indices["CAPTIONS"]))
        # for i in range (len(transformed_data_with_indices["CAPTIONS"])):
             
        #     if len(transformed_data_with_indices["CAPTIONS"][i]) >=  20 :
        #         print(transformed_data_with_indices["CAPTIONS"][i] )
       
        assert len( transformed_data_with_indices["IMAGES"])  == len(transformed_data_with_indices["CAPTIONS"])
        
        # print(transformed_data["CAPTIONS"][0])
        # print(transformed_data_with_indices["CAPTIONS"][0])

        return transformed_data_with_indices

    def save_data_to_json(data, file_path):
        """
        保存处理后的数据到 JSON 文件。
        """
        with open(file_path, 'w') as fw:
            json.dump(data, fw)


    def split_data_for_validation(transformed_data, validation_ratio=0.1,num_captions_per_image = 5):
        """
        从训练集中随机抽取一部分数据作为验证集。
        """
        # 按图像分组数据
        # grouped_data = [{"IMAGE": transformed_data["IMAGES"][i // num_captions_per_image], 
        #                  "CAPTIONS": transformed_data["CAPTIONS"][i:i + num_captions_per_image]} 
        #                 for i in range(0, len(transformed_data["CAPTIONS"]), num_captions_per_image)]
        
        grouped_data = [{"IMAGE": transformed_data["IMAGES"][i], 
                         "CAPTIONS": transformed_data["CAPTIONS"][i]} 
                        for i in range(0, len(transformed_data["CAPTIONS"]))]
        # print(grouped_data[0:2])
        # 随机打乱并分割数据
        random.shuffle(grouped_data)
        split_index = int(len(grouped_data) * validation_ratio)

        val_data = {"IMAGES": [], "CAPTIONS": []}
        train_data = {"IMAGES": [], "CAPTIONS": []}

        for group in grouped_data[:split_index]:
            val_data["IMAGES"].extend([group["IMAGE"]])
            val_data["CAPTIONS"].extend([group["CAPTIONS"]])

        for group in grouped_data[split_index:]:
            train_data["IMAGES"].extend([group["IMAGE"]])
            train_data["CAPTIONS"].extend([group["CAPTIONS"]])

        # print(train_data["IMAGES"][0])
        # print(train_data["CAPTIONS"][0])
        # print(val_data["IMAGES"][0])
        # print(val_data["CAPTIONS"][0])

        return train_data, val_data
    
    
    
    with open(os.path.join(file_path, "test_captions.json"), 'r') as file:
        test_data = json.load(file)
    
    with open(os.path.join(file_path, "train_captions.json"), 'r') as file:
        train_data = json.load(file)


    # 处理 test_captions_data 和 train_captions_data
    transformed_test_data = process_captions("test")
    transformed_train_data = process_captions("train")

    # 处理词典
    word_to_index_combined, unk_index = process_vocab(test_data, train_data, os.path.join(file_path, 'vocab.json'))

    # 转换为索引
    transformed_test_data_with_indices = convert_captions_to_indices(transformed_test_data, word_to_index_combined, unk_index)
    transformed_train_data_with_indices = convert_captions_to_indices(transformed_train_data, word_to_index_combined, unk_index)

    final_train_data, val_data = split_data_for_validation(transformed_train_data_with_indices)

    # 保存为 JSON
    save_data_to_json(transformed_test_data_with_indices, os.path.join(file_path, 'test_data.json'))
    save_data_to_json(final_train_data, os.path.join(file_path, 'train_data.json'))
    save_data_to_json(val_data, os.path.join(file_path, 'val_data.json'))

    # x=input()
    # print(process_punctuation(x))

if __name__ == '__main__':
    create_dataset(file_path,vocab_path,image_path)
    







import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

# 初始化ROUGE
rouge = Rouge()

def calculate_meteor_score(reference, hypothesis):
    """
    计算METEOR分数。

    参数:
    reference (str): 参考文本，通常是人类生成的描述。
    hypothesis (str): 模型生成的描述。

    返回:
    float: 计算出的METEOR分数。
    """
    nltk.download('punkt')  # 下载分词器所需数据
    nltk.download('omw-1.4')
    nltk.download('wordnet')  # METEOR需要WordNet数据
    # 这三个可以放在外面，无需每次调用都要检测下载
    reference_tokens = word_tokenize(reference)   # 分词
    hypothesis_tokens = word_tokenize(hypothesis)  # 分词
    return meteor_score([reference_tokens], hypothesis_tokens)

def calculate_rouge_l_score(reference, hypothesis):
    """
    计算ROUGE-L分数。

    参数:
    reference (str): 参考文本。
    hypothesis (str): 模型生成的描述。

    返回:
    float: 计算出的ROUGE-L分数。
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']

if __name__ == '__main__':
    reference_text = "The upper clothing has long sleeves, cotton fabric and solid color patterns. The neckline of it is v-shape. The lower clothing is of long length. The fabric is denim and it has solid color patterns. This lady also wears an outer clothing, with cotton fabric and complicated patterns. This female is wearing a ring on her finger. This female has neckwear."
    hypothesis_text = ""
    meteor_score = calculate_meteor_score(reference_text, hypothesis_text)
    rouge_l_score = calculate_rouge_l_score(reference_text, hypothesis_text)

    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE-L Score: {rouge_l_score}")
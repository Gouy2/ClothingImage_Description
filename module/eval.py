from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import json
import torch

def ids_to_words(indices_list, vocab_path):
    # 词汇表反向映射：从索引找到单词
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    reverse_vocab = {int(v): k for k, v in vocab.items()}

    sentences = []

    for indices in indices_list:
        # 转换索引到单词，并去除特殊标记
        words = [reverse_vocab.get(index, '') for index in indices if reverse_vocab.get(index, '') not in ['<start>', '<end>', '<pad>', '<unk>']]
        # 将单词列表拼接成句子
        sentence = ' '.join(words)
        sentences.append(sentence)

    return sentences



def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

def evaluate(data_loader, model, config):
    rouge = Rouge()

    model.eval()
    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})
    cpi = config.captions_per_image
    device = next(model.parameters()).device

    #打印data_loader的大小
    print("len(data_loader):",len(data_loader))

    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with torch.no_grad():
            # print("i:",i)
            if (i+1)%10 == 0:
                print(f"{i+1}/{len(data_loader)}")
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), config.beam_k, config.max_len+2)
            # 候选文本
            cands.extend([filter_useless_words(text, filterd_words) for text in texts])
            # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])

    print("cands:",cands[0])
    print("refs:",refs[0])

    converted_refs = [ids_to_words(ref, '../data/cloth/vocab.json').split() for ref in refs]
    converted_cands = [ids_to_words(cand, '../data/cloth/vocab.json').split() for cand in cands]
    
    # print("cands",cands[0])
    # 实际上，每个候选文本对应cpi条参考文本

    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
        
    # bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
        
    # print(cands)
    # print(len(cands))
    # print(refs)
    # print(len(cands))

    # 计算METEOR值
    scores = []
    for ref, cand in zip(converted_refs, converted_cands):
        cand_tokens = cand.split()  # 分词候选文本
        ref_tokens = [r.split() for r in ref]  # 分词每个参考文本
        score = meteor_score(ref_tokens, cand_tokens)
        scores.append(score)
    meteor = sum(scores) / len(scores)
    # 计算ROUGE值
    # rouge_score = rouge.get_scores(cands, refs, avg=True)

    model.train()
    # return bleu4
    return meteor 
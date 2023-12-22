import torch
import torch.nn as nn

from encoder import ViTEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, vocab, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.encoder = ViTEncoder()  # ViT 作为编码器
        self.decoder = TransformerDecoder(
            vocab, len(vocab), embed_dim, num_heads, num_layers, ff_dim, dropout
        )

    def forward(self, images, captions):
        image_code = self.encoder(images)
        return self.decoder(captions, image_code )
    
    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device

        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k,1,1,1)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            # 存储已生成完整的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k

            while True:
                preds, hidden_state = self.decoder(image_code, cur_sents, None) # 现在直接传递整个序列
                preds = preds[:, -1, :] # 只考虑最后一个输出
                preds = nn.functional.log_softmax(preds, dim=1)

                probs = probs.repeat(1, preds.size(1)) + preds
                values, indices = probs.view(-1).topk(k, 0, True, True)
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size

                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)

                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_sents = cur_sents[end_indices].tolist()
                    end_probs = values[end_indices]
                    k -= len(end_indices)
                    if k == 0:
                        break

                cur_indices = [idx for idx, word in enumerate(word_indices) if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1,1)

                if cur_sents.size(1) >= max_len:
                    break

            if len(end_sents) == 0:
                gen_sent = cur_sents[0].tolist()
            else: 
                gen_sent = end_sents[end_probs.index(max(end_probs))]

            texts.append(gen_sent)
        return texts


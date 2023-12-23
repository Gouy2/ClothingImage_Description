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
        # print("Captions shape:", captions.shape)
        # print("Image code shape:", image_code.shape)
        return self.decoder(captions, image_code )
    
    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device

        for image_code in image_codes:
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1)
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k

            end_sents = []  # 初始化空列表
            end_probs = []

            while True:
                preds = self.decoder(cur_sents, image_code)  # 不使用.squeeze(1)
                
                # 确保preds是三维的
                if preds.dim() == 2:
                    preds = preds.unsqueeze(1)

                preds = preds[:, -1, :]  # 获取序列中的最后一个词的预测
                preds = nn.functional.log_softmax(preds, dim=1)

                probs = probs.repeat(1, preds.size(1)) + preds
                values, indices = probs.view(-1).topk(k, 0, True, True)
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size

                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)

                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_sents.extend(cur_sents[end_indices].tolist())
                    end_probs.extend(values[end_indices].tolist())
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



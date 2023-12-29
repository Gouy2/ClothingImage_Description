import os
import json
import torch
import torch.nn as nn
from argparse import Namespace
import sys
import time
from tqdm.autonotebook import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from arctic import ARCTIC
from module.loss_opt import PackedCrossEntropyLoss,get_optimizer,adjust_learning_rate
from module.eval import evaluate
from module.dataset import create_dataset,mktrainval



# 设置模型超参数和辅助变量
config = Namespace(
    max_len = 120,
    captions_per_image = 1,
    batch_size = 16,
    image_code_dim = 2048,
    word_dim = 512,
    hidden_size = 512,
    attention_dim = 512, 
    num_layers = 1, 
    encoder_learning_rate = 0.0001,
    decoder_learning_rate = 0.0005,
    lr_update = 1, # 每隔多少个epoch，更新一次学习速率
    warmup_epochs = 5, # 前warmup_epochs个epoch，学习速率线性增长
    num_epochs = 10,
    grad_clip = 5.0, 
    alpha_weight = 1.0, 
    evaluate_step = 300, # 每隔多少步在验证集上测试一次
    checkpoint = None, # 如果不为None，则利用该变量路径的模型继续训练
    # checkpoint = './model/ckpt_model.ckpt', 
    best_checkpoint = './model/best_model.ckpt', # 验证集上表现最优的模型的路径
    last_checkpoint = './model/last_model.ckpt', # 训练完成时的模型的路径
    beam_k = 3
)


def main():
    # 设置GPU信息
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # 数据路径
    data_dir = '../data/cloth/'
    vocab_path = '../data/cloth/vocab.json'
    image_path = "../data/cloth/images"

    #加载数据
    create_dataset(data_dir, vocab_path ,image_path)

    print("数据加载完成")

    train_loader, valid_loader, test_loader = mktrainval(data_dir, vocab_path, config.batch_size)

    print("数据集划分完成")

    # 模型
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)



    # 随机初始化 或 载入已训练的模型
    start_epoch = 1
    checkpoint = config.checkpoint
    if checkpoint is None:
        model = ARCTIC(config.image_code_dim, vocab, config.word_dim, config.attention_dim, config.hidden_size, config.num_layers)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

    print("模型加载完成")

    # 优化器
    optimizer = get_optimizer(model, config)

    # 将模型拷贝至GPU，并开启训练模式
    model.to(device)
    model.train()

    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(device)

    best_res = 0
    print("开始训练")
    fw = open('log.txt', 'w')



    for epoch in range(start_epoch, config.num_epochs +start_epoch ):
        adjust_learning_rate(optimizer, epoch, config)
                             #.warmup_epochs, config.num_epochs, config.encoder_learning_rate, config.decoder_learning_rate)

        print('Epoch {}'.format(epoch))
        print('-' * 10)

        start_time = time.time()
        model.train()

        train_loss = 0
        total_steps = len(train_loader)

        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=total_steps)

        for i, (imgs, caps, caplens) in progress:
            optimizer.zero_grad()
            # 1. 读取数据至GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 2. 前馈计算
            predictions, alphas, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)

            # 3. 计算损失
            # captions从第2个词开始为targets
            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
            # 重随机注意力正则项，使得模型尽可能全面的利用到每个网格
            # 要求所有时刻在同一个网格上的注意力分数的平方和接近1
            loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()

            train_loss += loss.item()

            loss.backward()
            # 梯度截断
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # 4. 更新参数
            optimizer.step()
            
            progress.set_description("Loss: {:.4f}".format(loss.cpu()))

            if (i + 1) % 100 == 0:
                tqdm.write('epoch %d, step %d: loss=%.2f \n' % (epoch, i + 1, loss.cpu()))
                fw.write('epoch %d, step %d: loss=%.2f \n' % (epoch, i + 1, loss.cpu()))
                fw.flush()

            state = {
                    'epoch': epoch,
                    'step': i,
                    'model': model,
                    'optimizer': optimizer
                    }
            

        progress.close()  # 关闭进度条

        end_time = time.time()
        tqdm.write('TrainLoss: {:.3f} | Time Elapsed {:.3f} sec'.format(train_loss / total_steps, end_time - start_time))

        print("验证中...")
        meteor , rouge_score = evaluate(valid_loader, model, config)

        if best_res < meteor:
            best_res = meteor
            torch.save(state, config.best_checkpoint)

        torch.save(state, config.last_checkpoint)

        fw.write('Validation@epoch, %d, METEOR=%.2f \n' % 
            (epoch,  meteor))
        fw.write('Validation@epoch, %d, ROUGE=%.2f \n' %
            (epoch, rouge_score))
                
        fw.flush()

        print('Validation@epoch, %d, METEOR=%.2f' % 
            (epoch, meteor))
        print('Validation@epoch, %d, ROUGE=%.2f' %
            (epoch, rouge_score))
               
                
    checkpoint = torch.load(config.best_checkpoint)

    model = checkpoint['model']

    meteor = evaluate(test_loader, model, config)

    print("Evaluate on the test set with the model that has the best performance on the validation set")

    print('Epoch: %d, METEOR=%.2f' % 
        (checkpoint['epoch'], meteor))
    
    print('Epoch: %d, ROUGE=%.2f' % 
        (checkpoint['epoch'], rouge_score))
    
    fw.write('Epoch: %d, METEOR-4=%.2f' % 
        (checkpoint['epoch'], meteor))
    fw.close()

      

if __name__ == '__main__':
    main()
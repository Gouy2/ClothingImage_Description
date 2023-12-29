import os
import json
import torch
import torch.nn as nn
import sys
from argparse import Namespace
import time
from tqdm.autonotebook import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from module.dataset import create_dataset,mktrainval
from module.loss_opt import TransformerCrossEntropyLoss,get_optimizer,adjust_learning_rate
from module.eval import evaluate
from integrate import Transformer


# 设置模型超参数和辅助变量
config = Namespace(
    max_len = 120,
    captions_per_image = 1,
    # image_code_dim = 2048,
    batch_size = 16,
    word_dim = 512,
    num_heads = 8 , #注意力头数
    num_layers = 6 , #解码器中的层数
    ff_dim = 1024 , #前馈网络的维度
    encoder_learning_rate = 0.0001,
    decoder_learning_rate = 0.0005,
    num_epochs = 10,
    grad_clip = 5.0, 
    alpha_weight = 1.0, 
    evaluate_step = 600, # 每隔多少步在验证集上测试一次
    checkpoint = None, # 如果不为None，则利用该变量路径的模型继续训练
    # checkpoint = './model/ckpt_model.ckpt', 
    best_checkpoint = './model/best_model.ckpt', # 验证集上表现最优的模型的路径
    last_checkpoint = './model/last_model.ckpt', # 训练完成时的模型的路径
    beam_k = 3
)


def main():
    # 设置GPU信息
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # # torch.cuda.device_count()
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
        model = Transformer( vocab, config.word_dim, config.num_heads, config.num_layers, config.ff_dim)
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
    loss_fn = TransformerCrossEntropyLoss().to(device)

    best_res = 0
    print("开始训练")
    fw = open('log.txt', 'w')


    for epoch in range(start_epoch, config.num_epochs):
        adjust_learning_rate(optimizer, epoch, config)

        print('Epoch {}'.format(epoch))
        print('-' * 10)

        start_time = time.time()
        model.train()

        train_loss = 0
        total_steps = len(train_loader)

        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=total_steps)

        for i, (imgs, caps, caplens) in progress:
            
            # 1. 读取数据至GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            
            # 2. 前馈计算

            predictions = model(imgs, caps)

            caplens = caplens.to('cpu').long()  # 确保长度在 CPU 上并且为 int64 类型

            # print("caps[:, 1:]:",caps[:, 1:])

            # 3. 计算损失
            # captions从第2个词开始为targets
            caplens_adjusted = caplens - 1
            loss = loss_fn(predictions, caps[:, 1:], caplens_adjusted)


            # 4. 反向传播和优化
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 反向传播计算梯度

            # 梯度截断
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()       # 更新参数

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

        eval_score = meteor + rouge_score

        if best_res < eval_score:
            best_res = eval_score
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
import sys
import json
import torch
from PyQt5.QtWidgets import QApplication
from torchvision import transforms
import sys
# sys.path.append("./gru") #model定义路径 
sys.path.append("./trsfm") #model定义路径
from module.gui import ImageCaptioningApp



if __name__ == '__main__':

    vocab_path='./data/cloth/vocab.json'
    # model = '../save/gru/3_1.ckpt' 
    model = '../save/trs/t1-1.ckpt' 
    #model = './model/_model.ckpt'
    
    app = QApplication(sys.argv)
    ex = ImageCaptioningApp(model,vocab_path)
    ex.show()
    sys.exit(app.exec_())
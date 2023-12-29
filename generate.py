import sys
import json
import torch
from PyQt5.QtWidgets import QApplication
from torchvision import transforms
import sys
sys.path.append("./gru") 
# sys.path.append("./trsfm") 
from module.gui import ImageCaptioningApp



if __name__ == '__main__':

    vocab_path='./data/cloth/vocab.json'

    model = '../save/gru/best_model.ckpt'  #Resnet+GRU

    # model = '../save/trs/best_model.ckpt'  #Vit+Transformer
    
    app = QApplication(sys.argv)
    ex = ImageCaptioningApp(model,vocab_path)
    ex.show()
    sys.exit(app.exec_())
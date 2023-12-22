import sys
import json
import torch
from PyQt5.QtWidgets import QApplication
from torchvision import transforms
import sys
sys.path.append("D:/NewCode/Cloth/ClothingImage_Description/gru") #output.py绝对路径 Gy
# #sys.path.append("/ClothingImage_Description/gru") #绝对路径
# #sys.path.append("/ClothingImage_Description/gru") #绝对路径
from module.gui import ImageCaptioningApp


if __name__ == '__main__':

    with open('./data/cloth/vocab.json', 'r') as f:
        vocab = json.load(f)

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = '../save/3_1.ckpt' #Gy
    #model = './model/_model.ckpt'
    
    app = QApplication(sys.argv)
    ex = ImageCaptioningApp(model,vocab,transform)
    ex.show()
    sys.exit(app.exec_())
import sys
import json
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from torchvision import transforms

from gru.output import generate_caption, indices_to_sentence_nested


class ImageCaptioningApp(QWidget):
    def __init__(self, model, vocab, transform):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.transform = transform
        self.initUI()

    def initUI(self):
        # 设置窗口标题和初始大小
        self.setWindowTitle('图像描述生成')
        self.resize(800, 600)

        # 主垂直布局
        main_layout = QVBoxLayout(self)

        # 上部水平布局用于放置按钮和图片
        top_layout = QHBoxLayout()

        # 垂直布局用于放置按钮
        button_layout = QVBoxLayout()

        # 创建上传图片按钮
        upload_btn = QPushButton('上传图片', self)
        upload_btn.setFixedSize(120, 40)  # 设置按钮大小
        upload_btn.clicked.connect(self.uploadImage)
        button_layout.addWidget(upload_btn)
        button_layout.setAlignment(upload_btn, Qt.AlignHCenter)  # 按钮居中对齐

        # 创建生成描述按钮
        generate_btn = QPushButton('生成描述', self)
        generate_btn.setFixedSize(120, 40)  # 设置按钮大小
        generate_btn.clicked.connect(self.generateCaption)
        button_layout.addWidget(generate_btn)
        button_layout.setAlignment(generate_btn, Qt.AlignHCenter)  # 按钮居中对齐

        # 将按钮布局添加到上部布局
        top_layout.addLayout(button_layout)

        # 创建和设置图片显示标签
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 450)
        top_layout.addWidget(self.image_label)

        # 将上部布局添加到主布局
        main_layout.addLayout(top_layout)

        # 创建和设置描述显示标签
        self.caption_label = QLabel('Caption will appear here...', self)
        self.caption_label.setObjectName("DescriptionLabel")  # 设置一个唯一的对象名称
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)
        main_layout.addWidget(self.caption_label)

        # 应用样式表
        self.applyStyleSheet()

        # 设置主布局
        self.setLayout(main_layout)

    def applyStyleSheet(self):
        self.setStyleSheet("""
            QWidget { 
                background-color: #f0f0f0;
                font-family: 'Arial';
            }
            QLabel {
                /* 移除font-size，使其不影响所有QLabel */
                color: #333;
            }
            QPushButton {
                background-color: #5c5c5c;
                color: white;
                border-radius: 10px;
                padding: 12px 20px;
                font-size: 16px;  /* 仅设置按钮内的文字大小 */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6c6c6c;
            }
            #ImageLabel, #CaptionLabel {
                border: 2px solid #5c5c5c;
                border-radius: 10px;
            }
            #DescriptionLabel {  /* 仅针对描述标签的样式 */
                font-size: 18px;  /* 增大描述标签的字体大小 */
                padding: 10px;  /* 增加描述标签的内边距 */
                color: #444;
            }
        """)

        # 设置图片和描述标签的特定样式
        self.image_label.setObjectName("ImageLabel")
        self.caption_label.setObjectName("DescriptionLabel")
        
        



    def uploadImage(self):
        """上传图片并在 GUI 中显示"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "JPEG Files (*.jpg);;PNG Files (*.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def generateCaption(self):
        """生成图片描述并显示"""
        if hasattr(self, 'image_path'):
            caption = generate_caption(self.image_path, self.model, self.transform)
            caption_words = indices_to_sentence_nested(caption, self.vocab)
            self.caption_label.setText(caption_words)
            print("图片描述:", caption_words)



if __name__ == '__main__':


    with open('../data/cloth/vocab.json', 'r') as f:
        vocab = json.load(f)

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #model = './model/_model.ckpt'
    model = '.../save/3_1.ckpt'


    app = QApplication(sys.argv)
    ex = ImageCaptioningApp(model,vocab,transform)
    ex.show()
    sys.exit(app.exec_())

   

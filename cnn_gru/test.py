import torch
# print(torch.__version__)  #注意是双下划线
print(torch.cuda.is_available())

print(torch.cuda.device_count()) #可用GPU数量
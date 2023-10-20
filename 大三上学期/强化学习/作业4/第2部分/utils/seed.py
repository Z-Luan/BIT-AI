import torch
import numpy as np
import random
import os

# 调用种子函数
def seed(seednum = 1):
    random.seed(seednum)
    np.random.seed(seednum)
    # 设置 CPU 生成随机数的种子
    torch.manual_seed(seednum)
    # 设置 GPU 生成随机数的种子
    torch.cuda.manual_seed(seednum)
    os.environ['PYTHONHASHSEED'] = str(seednum)
    # True 每次返回的卷积算法确定, 即默认算法
    # 如果 Torch 的随机种子为固定值, 可以保证神经网络的输入相同时输出也一致
    torch.backends.cudnn.deterministic = True
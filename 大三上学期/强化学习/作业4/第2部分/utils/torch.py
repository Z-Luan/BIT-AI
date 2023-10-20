import torch

# 设置 Torch 属性
class TorchHelper:
    
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def f(self, x):
        return torch.tensor(x).float().to(self.device)
    
    def i(self, x):
        return torch.tensor(x).int().to(self.device)
    
    def l(self, x):
        return torch.tensor(x).long().to(self.device)
import torch
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
# 如果是 Mac M 芯片，可以打印 torch.backends.mps.is_available()

# 生成一个随机张量
x = torch.rand(3, 4)
print(f"Random Tensor:\n{x}")
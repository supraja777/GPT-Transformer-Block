import torch
import torch.nn as nn

linear = nn.Linear(3, 2) # input dim = 3, output dim = 2

x = torch.tensor([1.0, 2.0, 3.0])
y = linear(x)
print(y)
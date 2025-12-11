import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gelu_activation import GELU

# Unlike ReLU, which outputs zero for any negative input, 
# GELU allows for a small, non-zero output for negative values. 

# The smoothness of GELU can lead to better optimization properties during training, 
# as it allows for more nuanced adjustments to the model’s parameters. 
# In contrast, ReLU has a sharp corner at zero (figure 4.18, right), which can 
# sometimes make optimization harder, especially in networks that are very deep or 
# have complex architectures.

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)     #1
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
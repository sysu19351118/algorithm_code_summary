import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNLayers(nn.Module):
    def __init__(self, dim, num_layers):
        super(DCNLayers, self).__init__()
        self.layers = nn.ModuleList([ 
            nn.Linear(dim, dim, bias = True)  for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = layer(x) * x0 + x
        return x


if __name__ == "__main__":
    a = torch.zeros((2,15,64))
    model = DCNLayers(64,3)
    print(model(a).shape)
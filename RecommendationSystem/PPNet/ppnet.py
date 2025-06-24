import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, sigmoid_at_last = False, dropout_prob=0.2):
        super().__init__()
        mlp_depth = len(input_dim)
        assert len(input_dim) == len(output_dim)
        layers = []
        for i in range(mlp_depth):
            layers.append(
                nn.Linear(input_dim[i], output_dim[i])
            )    
            if i != (mlp_depth-1):
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout_prob))
        self.if_sigmoid_last = sigmoid_at_last
        self.layers = nn.Sequential(*layers)

        if self.if_sigmoid_last:
            self.sigmoid_layer = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layers(x)
        if self.if_sigmoid_last:
            x = 2*self.sigmoid_layer(x)
        
        return x

class PPNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.sku_embedding_layers = MLP([128,128], [128,64])
        self.user_embedding_layers = MLP([128,128], [128,64], sigmoid_at_last=True)

        self.sku_embedding_layers_2 = MLP([64, 64], [64, 64])
        self.user_embedding_layers_2 = MLP([128, 64], [64, 64], sigmoid_at_last=True)

    
    def forward(self, user_emb, sku_emb):
        x_sku = self.sku_embedding_layers(sku_emb)
        x_user = self.user_embedding_layers(user_emb)

        x_sku = x_sku * x_user

        x_sku_2 = self.sku_embedding_layers_2(x_sku)
        x_user_2 = self.user_embedding_layers_2(user_emb)

        return x_sku_2 * x_user_2




if __name__ == "__main__":
    model = PPNet()
    a = torch.zeros((4,128))
    print(model(a,a).shape)
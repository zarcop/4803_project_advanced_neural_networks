import torch
from torch import nn

class Embedding_Layer(nn.Module):

    def __init__(self, in_channels, patch_size, embedded_dim) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embedded_dim, kernel_size= patch_size, stride = patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

class Transformer_Block(nn.Module):

    def __init__(self, embedded_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm_layer1 = nn.LayerNorm(embedded_dim)
        self.attention_block = nn.MultiheadAttention(embedded_dim, num_heads, dropout, batch_first= True)
        self.norm_layer2 = nn.LayerNorm(embedded_dim)
        #hyperparameter of the FNN connected layers of the transfomer block.
        hidden_dimensions = int(embedded_dim * mlp_ratio)
        self.mlp_block = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dimensions),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions, embedded_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):

        #literally from the medium paper

        attention_result, _ = self.attention_block(self.norm_layer1(x), self.norm_layer1(x), self.norm_layer1(x))
        # embedding  + attention(norm(embedding))
        x = x + attention_result\
        # embedding + attention(norm(embedding)) + mlp (norm(embedding))
        x = x + self(self.norm_layer2(x))
        return x


    



"""
Vanilla implementation of transformers with separate blocks for patch embedding and a transformer block using multihead attention
and residual multilayer perceptrons.
the hyperparameters that are going to be tuned in validation/finetuning:
num_heads,
mlp_ratio
drouout
"""

class Vision_Transformers(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, k_classes, embedded_dim, depth, num_heads, mlp_ratio, dropout):
        self.patch_embedding = Embedding_Layer(in_channels, patch_size, embedded_dim)
        num_patches = (img_size// patch_size) ** 2
        # adding the class tokens to the embeddings
        self.class_token = nn.Parameter(torch.zeros(1,1,embedded_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, embedded_dim))
        self.positional_dropout = nn.Dropout(dropout)
        # Stack of Transformer Blocks
        self.attention_blocks = nn.Sequential(*[Transformer_Block(embedded_dim, num_heads, mlp_ratio, dropout) for layers in range(depth)])
        self.norm = nn.LayerNorm(embedded_dim)
        #final layer using FNN to classify the embedding into the classes
        self.head = nn.Linear(embedded_dim, k_classes)

    def forward(self,x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        #expand the class tokens to the batch size and prepend them to the image patches
        class_tokens = self.class_token.expand(B,-1,-1)
        x = torch.cat((class_tokens, x), dim = 1)
        x = x + self.positional_encoding(x)
        #dropout to reduce the dimensions:
        x = self.positional_dropout(x)
        # transformer encoding
        x = self.attention_blocks(x)
        #final layer norm
        x = self.norm(x)
        # extract the class token that was prepended at the beginning of the forward pass.
        class_out = x[:,0]
        output = self.head(class_out)
        return output









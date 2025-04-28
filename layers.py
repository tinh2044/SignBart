import torch
from torch import nn
import math

class PositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):

        bsz, seq_len = inputs_embeds.shape[:2]
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1).to(inputs_embeds.device)
        positions_embeddings = super().forward(positions + self.offset)

        return positions_embeddings

class BartLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        self.dropout =dropout

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training) 
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

class Projection(nn.Module):
    def __init__(self, config):
        super(Projection, self).__init__()
        
        self.proj_x1 = nn.Linear(len(config['joint_idx']), config['d_model'])
        self.proj_y1 = nn.Linear(len(config['joint_idx']), config['d_model'])

    def forward(self, inputs):
        x_coord = inputs[:, :, :, 0]
        y_coord = inputs[:, :, :, 1]
        
        x_embed = self.proj_x1(x_coord)       

        y_embed = self.proj_y1(y_coord)        
        
        return x_embed, y_embed
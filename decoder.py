import torch
import torch.nn as nn
from .attention import CrossAttention, CausalSelfAttention
from .layers import PositionalEmbedding
from utils import create_attention_mask, create_causal_attention_mask

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']

        self.self_attn = CausalSelfAttention(
            d_model=self.d_model,
            num_heads=config['decoder_attention_heads'],
            dropout=config['attention_dropout'])
        
        self.dropout = config['dropout']
        self.activation_fn = nn.GELU()
        self.activation_dropout = config['activation_dropout']

        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.encoder_attn = CrossAttention(
            self.d_model,
            config['decoder_attention_heads'],
            dropout=config['attention_dropout']
        )
        
        self.encoder_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, config['decoder_ffn_dim'])
        self.fc2 = nn.Linear(config['decoder_ffn_dim'], self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        residual = hidden_states
    
        hidden_states = self.self_attn(hidden_states, attention_mask)
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        hidden_states  = self.encoder_attn(hidden_states, encoder_hidden_states, encoder_attention_mask)
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.layerdrop = config['decoder_layerdrop'] 
        embed_dim = config['d_model']
        
        self.embed_positions = PositionalEmbedding(
            config['max_position_embeddings'],
            embed_dim,
        )
            
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config['decoder_layers'])])

        self.layernorm_embedding = nn.LayerNorm(config['d_model'])

    def forward(self, encoder_hidden_states, encoder_attention_mask, y_embed, attention_mask):
        input_shape = y_embed.size()[:-1]

        attention_mask = create_causal_attention_mask(attention_mask, input_shape, y_embed)
       
        encoder_attention_mask = create_attention_mask(
            encoder_attention_mask, y_embed.dtype, tgt_len=input_shape[-1]
        )

        pos_embed = self.embed_positions(y_embed)
        y_embed = y_embed + pos_embed
        hidden_states = self.layernorm_embedding(y_embed)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
                )

        return hidden_states
import torch
from torch import nn

class BaseAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout = 0.0, bias = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads

        if (self.head_dim * num_heads) != self.d_model:
            raise ValueError(
                f"d_model must be divisible by num_heads (got `d_model`: {self.d_model}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

class SelfAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__(d_model, num_heads, dropout=0.0, bias=True)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads

        if (self.head_dim * num_heads) != d_model:
            raise ValueError("d_model must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, hidden_states, attention_mask):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output

class CrossAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__(d_model, num_heads, dropout=0.0, bias=True)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads

        if (self.head_dim * num_heads) != d_model:
            raise ValueError("d_model must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        

    def forward(self, hidden_states, key_value_states, attention_mask):
        bsz, tgt_len, _ = hidden_states.size()
        src_len = key_value_states.size(1)

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output
    
class CausalSelfAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__(d_model, num_heads, dropout=0.0, bias=True)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads

        if (self.head_dim * num_heads) != d_model:
            raise ValueError("d_model must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, hidden_states, attention_mask):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=hidden_states.device)).view(1, 1, tgt_len, tgt_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output

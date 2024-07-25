from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024      # maximum sequence length
    vocab_size: int = 50257     # number of tokens: 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12           # number of layers
    n_head: int = 12            # number of heads
    n_embd: int = 768           # embedding dimension


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following OpenAI Hugging Face naming convention
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()      # B: batch size, T: sequence length, C: embedding dimensionality (n_embd)

        # nh: number of heads, hs: head size and, C (number of channels) = nh * hs 
        # e.g. in GPT-2 (124M), n_head=12, hs=64 so, nh*hs=C=768 channels in the transformer 

        # calculate query, key, value for all heads in batch and move head to be a batch dimension
        qkv = self.attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)

        # attention weights (materializes the large (T, T) matrix for all the queries and keys)
        att = q @ k.transpose(-2, -1) * (1.0 / k.size(-1)**0.5)
        att = torch.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v                                                             # (B, nh, T, hs)
        # re-assemble all heads outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()                                                     
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)                 # diverging layer
        self.gelu = nn.GELU(approximate='tanh')                                 # gaussian error linear unit
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)               # converging layer

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)                                 # layer normalization 1
        self.attn = CausalSelfAttention(config)                                 # self attention block
        self.ln_2 = nn.LayerNorm(config.n_embd)                                 # layer normalization 2
        self.mlp = MLP(config)                                                  # feed forward network

    def forward(self, x):
        # skip connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),               # weights for token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),               # weights for positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),         # hidden blocks
            ln_f = nn.LayerNorm(config.n_embd),                                 # layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language modelling head

    @classmethod
    def from_pretrained(cls, model_type):
        """loads pre-trained GPT-2 model weights from Hugging Face"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print('loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M parameters
        }[model_type]
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints
        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints

        # create a from-scratch initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard mask buffer, not a parameter

        # initialize a Hugging Face transformers model
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and, match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]        # discard mask, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard mask, just a buffer

        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}' 

        # basically, the openAI checkpoints use a 'Conv1D' module, but we want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_attn.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights, we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

model_type = 'gpt2'
model = GPT.from_pretrained(model_type)
print(f'successfully loaded weights from pretrained gpt: {model_type}')
        
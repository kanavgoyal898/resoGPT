from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken

# --------------------------------------------------------------------------------------------------------------------------------

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
        self.c_proj.RESOGPT_SCALE_INIT = True
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
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)

        # attention weights (materializes the large (T, T) matrix for all the queries and keys)
        att = q @ k.transpose(-2, -1) * (1.0 / k.size(-1)**0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
        self.c_proj.RESOGPT_SCALE_INIT = True

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # hidden blocks
            ln_f = nn.LayerNorm(config.n_embd),                                 # layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language modelling head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # parameter initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # weight initialization std=0.02 according to OpenAI GPT-2 implementation
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'RESOGPT_SCALE_INIT'):
                std /= (2 * config.n_layer)**0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f'cannot forward sequence of length {T}, block size is only {self.config.block_size}'

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # (T, ): positions shape
        pos_emb = self.transformer.wpe(pos)                             # (T, n_embd): position embeddings shape
        tok_emb = self.transformer.wte(idx)                             # (B, T, n_embd): token embeddings shape
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                        # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
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
    
# --------------------------------------------------------------------------------------------------------------------------------

class DataLoader:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load tokens from disk and store them in memory
        with open('tiny_shakespeare.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding(model_type)
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens (i.e., {len(self.tokens) // (B*T)} batches per epoch)')

        # state
        self.current_position = 0

    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + (self.B*self.T) + 1]
        x = buf[:-1].view(self.B, self.T)   # inputs
        y = buf[1:].view(self.B, self.T)    # targets
        # advance the position in the tensor
        self.current_position += self.B*self.T
        # if the next batch would be out of bounds, reset
        if self.current_position + (self.B*self.T) + 1 > len(self.tokens):
            self.current_position = 0
        return x, y
    
# --------------------------------------------------------------------------------------------------------------------------------

model_type = 'gpt2'

# auto-device detection
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

config = GPTConfig()
model = GPT(config)
train_loader = DataLoader(4, 32)

model.eval()
model.to(device)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i+1:2d}, loss: {loss.item():.4f}')

import sys
sys.exit(0)

# generate
torch.manual_seed(2147483647)
torch.mps.manual_seed(2147483647)

max_length = 24
num_return_sequences = 8

while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits = model(x)                                       # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]                               # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # top-50 sampling (Hugging Face pipeline default)
        # topk_probs here becomes (num_return_sequences, k), topk_indices here becomes (num_return_sequences, k)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)                   # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)               # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded, '...\n')

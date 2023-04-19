import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters

batch_size = 64
block_size = 256
learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 500
max_iters = 10001
n_embd = 384
n_head = 6
n_layer =6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# mapping from characters to ints
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# encoding dataset and putting it into torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


xb, yb = get_batch("train")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

class LayerNorm:
    def __init__(self, dim, momentum, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones((1, dim))
        self.beta = torch.zeros((1, dim))

    def __call__(self, x):
        x_mean = x.mean(1, keepdim=True)
        x_var = x.var(1, keepdim=True)

        x_hat = (x - x_mean) / (x_var + self.eps)  # normalization to unit variance
        self.out = self.gamma * x_hat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.values(x)

        wei = keys @ queries.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ values
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# The model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)


        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx is (B,T) tensor
        # logits is (B,T,C) tensor
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx  is (B,T) array of current context
        # we whant to make it an (B,T+max_new_tokens) tenosir
        for _ in range(max_new_tokens):
            # crop idx  to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            # take only the last timestep
            logits = logits[:, -1, :]
            # getting probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution of probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to a sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()

m = model.to(device)

# Traning
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    # once in a eval_interval steps evaluate train and test loss
    if step % eval_interval == 0:
        losses = estimate_loss()

        print(f"step {step}  train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # get a batch
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
estimate_loss()

# Generation from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

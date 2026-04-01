import numpy as np
import torch
import torch.nn as nn

SEED = 42
EPOCHS = 60
BATCH = 256
LR = 3e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 32
N_HEAD = 2
N_LAYER = 2
FF_DIM = 64
SEQ_LEN = 17
VOCAB = 5
N_ACTIONS = 4


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(SEQ_LEN, D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_ACTIONS)

    def forward(self, x):
        side = x[:, 16] + 2
        toks = torch.cat([x[:, :16], side[:, None]], dim=1)
        h = self.token_emb(toks) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h[:, -1])
        return self.head(h)


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def make_model():
    model = TinyPolicy()
    return model


def to_state_dict_numpy(model):
    out = {}
    for key, value in model.state_dict().items():
        out[key] = value.detach().cpu().numpy().copy()
    return out


def load_model(model_dict):
    model = make_model()
    state = model.state_dict()
    loaded = {}
    for key, value in state.items():
        loaded[key] = torch.from_numpy(model_dict[key]).to(dtype=value.dtype)
    model.load_state_dict(loaded)
    model.eval()
    return model


def train(X_train, y_train):
    set_seed()
    X = torch.from_numpy(X_train.astype(np.int64))
    y = torch.from_numpy(y_train.astype(np.int64))

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for _ in range(EPOCHS):
        perm = torch.randperm(X.shape[0])
        for start in range(0, X.shape[0], BATCH):
            idx = perm[start:start + BATCH]
            xb = X[idx]
            yb = y[idx]
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return to_state_dict_numpy(model)


@torch.no_grad()
def predict(model, X):
    mod = load_model(model)
    xb = torch.from_numpy(X.astype(np.int64))
    logits = mod(xb)
    return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

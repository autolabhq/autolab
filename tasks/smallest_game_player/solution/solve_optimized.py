"""
Reference solution: per-action scoring with weight sharing.

Key insight: reformulate 4-class prediction as action scoring.
For each of 4 columns, compute 36 tactical features, then apply a
shared Linear(36→H)→ReLU→Linear(H→1) to get a scalar score.
Argmax over 4 scores gives the predicted column.

Weight sharing across actions means params scale with feature count,
not with num_actions: 36*24 + 24 + 24*1 + 1 = 913 params.
"""
import numpy as np
import torch

SEED = 42
H = 24
EPOCHS = 400
LR = 1e-2
WEIGHT_DECAY = 1e-4

LINES = []
for r in range(4):
    for c in range(2):
        LINES.append([r * 4 + c, r * 4 + c + 1, r * 4 + c + 2])
for c in range(4):
    for r in range(2):
        LINES.append([r * 4 + c, (r + 1) * 4 + c, (r + 2) * 4 + c])
for r in range(2):
    for c in range(2):
        LINES.append([r * 4 + c, (r + 1) * 4 + c + 1, (r + 2) * 4 + c + 2])
for r in range(2):
    for c in range(2, 4):
        LINES.append([r * 4 + c, (r + 1) * 4 + c - 1, (r + 2) * 4 + c - 2])

CENTER = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def winmask(board, side):
    h = (board != 0).sum(0)
    out = np.zeros(4, dtype=np.float32)
    for c in range(4):
        if h[c] >= 4:
            continue
        b = board.copy()
        b[h[c], c] = side
        flat = b.reshape(-1)
        for line in LINES:
            if np.all(flat[line] == side):
                out[c] = 1.0
                break
    return out


def line_class_counts(flat, me, opp):
    cnt = np.zeros(8, dtype=np.float32)
    for line in LINES:
        vals = flat[line]
        m = np.sum(vals == me)
        o = np.sum(vals == opp)
        e = np.sum(vals == 0)
        cnt[0] += float(m == 3)
        cnt[1] += float((m == 2) and (e == 1) and (o == 0))
        cnt[2] += float((m == 1) and (e == 2) and (o == 0))
        cnt[3] += float((m == 0) and (e == 3))
        cnt[4] += float(o == 3)
        cnt[5] += float((o == 2) and (e == 1) and (m == 0))
        cnt[6] += float((o == 1) and (e == 2) and (m == 0))
        cnt[7] += float((m > 0) and (o > 0))
    return cnt


def featurize_actions(X):
    """Return (N, 4, 36) array — one 36-dim feature vector per action."""
    X = np.asarray(X)
    N = X.shape[0]
    F = np.zeros((N, 4, 36), dtype=np.float32)
    for i in range(N):
        b = X[i, :16].reshape(4, 4).copy()
        cur = int(X[i, 16])
        opp = 3 - cur
        h = (b != 0).sum(0)
        legal = (h < 4).astype(np.float32)
        cur_now = winmask(b, cur)
        opp_now = winmask(b, opp)
        global_now = line_class_counts(b.reshape(-1), cur, opp)
        for c in range(4):
            f = [1.0, legal[c], CENTER[c], 1.0 - CENTER[c], float(h[c]), float(4 - h[c])]
            f += list(h.astype(np.float32))          # 4: all column heights
            f += list(legal)                          # 4: all column legals
            f += [cur_now[c], opp_now[c], cur_now.sum(), opp_now.sum()]
            if legal[c]:
                b2 = b.copy()
                r = int(h[c])
                b2[r, c] = cur
                cur2 = winmask(b2, cur)
                opp2 = winmask(b2, opp)
                g2 = line_class_counts(b2.reshape(-1), cur, opp)
                f += [float(r), float(r * 4 + c), cur2.sum(), opp2.sum(),
                      cur_now[c], opp_now[c]]
                f += list(g2[[0, 1, 2, 4, 5, 6]])          # 6: line counts after move
                f += list((g2 - global_now)[[0, 1, 2, 4, 5, 6]])  # 6: delta line counts
            else:
                f += [0.0] * 18
            F[i, c] = np.asarray(f, dtype=np.float32)
    return F


def train(X_train, y_train):
    set_seed()
    F = torch.from_numpy(featurize_actions(X_train))
    y = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    net = torch.nn.Sequential(
        torch.nn.Linear(36, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        logits = net(F).squeeze(-1)   # (N, 4)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return {
        'w1': net[0].weight.detach().cpu().numpy().copy(),
        'b1': net[0].bias.detach().cpu().numpy().copy(),
        'w2': net[2].weight.detach().cpu().numpy().copy(),
        'b2': net[2].bias.detach().cpu().numpy().copy(),
    }


def predict(model, X):
    F = featurize_actions(X)
    h = np.maximum(0.0, F @ model['w1'].T + model['b1'])
    logits = (h @ model['w2'].T + model['b2']).squeeze(-1)
    return logits.argmax(axis=1).astype(np.int64)

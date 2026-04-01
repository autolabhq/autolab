#!/usr/bin/env python3
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, "/app")
import solve  # noqa: E402

ROWS = 4
COLS = 4
CONNECT = 3
MIN_ACC = 0.75


def winning_lines():
    lines = []
    for r in range(ROWS):
        for c in range(COLS - CONNECT + 1):
            lines.append(tuple(r * COLS + c + i for i in range(CONNECT)))
    for c in range(COLS):
        for r in range(ROWS - CONNECT + 1):
            lines.append(tuple((r + i) * COLS + c for i in range(CONNECT)))
    for r in range(ROWS - CONNECT + 1):
        for c in range(COLS - CONNECT + 1):
            lines.append(tuple((r + i) * COLS + (c + i) for i in range(CONNECT)))
    for r in range(ROWS - CONNECT + 1):
        for c in range(CONNECT - 1, COLS):
            lines.append(tuple((r + i) * COLS + (c - i) for i in range(CONNECT)))
    return lines


LINES = winning_lines()


def has_win(board, player):
    for a, b, c in LINES:
        if board[a] == board[b] == board[c] == player:
            return True
    return False


def legal_moves(board):
    moves = []
    for col in range(COLS):
        for row in range(ROWS):
            if board[row * COLS + col] == 0:
                moves.append(col)
                break
    return moves


def apply_move(board, player, col):
    nxt = list(board)
    for row in range(ROWS):
        idx = row * COLS + col
        if nxt[idx] == 0:
            nxt[idx] = player
            return tuple(nxt)
    raise ValueError


@lru_cache(maxsize=None)
def solve_state(board, player):
    if has_win(board, 1) or has_win(board, 2):
        return -1, -1
    moves = legal_moves(board)
    if not moves:
        return 0, -1
    best_score = -2
    best_move = -1
    for move in moves:
        nxt = apply_move(board, player, move)
        if has_win(nxt, player):
            score = 1
        else:
            opp_score, _ = solve_state(nxt, 3 - player)
            score = -opp_score
        if score > best_score or (score == best_score and move < best_move):
            best_score = score
            best_move = move
    return best_score, best_move


def enumerate_states():
    stack = [(tuple([0] * 16), 1)]
    seen = set()
    states = []
    while stack:
        board, player = stack.pop()
        if (board, player) in seen:
            continue
        seen.add((board, player))
        if has_win(board, 1) or has_win(board, 2):
            continue
        moves = legal_moves(board)
        if not moves:
            continue
        _, best = solve_state(board, player)
        states.append((np.asarray(board + (player,), dtype=np.int8), best))
        for move in moves:
            nxt = apply_move(board, player, move)
            if not has_win(nxt, player):
                stack.append((nxt, 3 - player))
    return states


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def model_checksum(model):
    total = 0.0
    for key in sorted(model):
        arr = model[key].astype(np.float64).ravel()
        total += float(np.dot(arr, np.arange(1, len(arr) + 1, dtype=np.float64)))
    return int(abs(total) * 1e3) % (10 ** 12)


def main():
    # Use the same pre-generated data as test.sh for consistency
    X_train = np.load("/app/X_train.npy")
    y_train = np.load("/app/y_train.npy")
    X_val = np.load("/tests/X_test.npy")
    y_val = np.load("/tests/y_test.npy")

    model = solve.train(X_train, y_train)
    if not isinstance(model, dict):
        fail(f"train() must return dict, got {type(model)}")
    for key, value in model.items():
        if not isinstance(key, str):
            fail(f"model key must be str, got {type(key)}")
        if not isinstance(value, np.ndarray):
            fail(f"model['{key}'] must be np.ndarray, got {type(value)}")

    preds = solve.predict(model, X_val)
    if not isinstance(preds, np.ndarray):
        fail(f"predict() must return np.ndarray, got {type(preds)}")
    if preds.shape != (len(X_val),):
        fail(f"predict() shape {preds.shape}, expected {(len(X_val),)}")
    if preds.dtype not in (np.int32, np.int64):
        fail(f"predict() dtype {preds.dtype}, expected int32 or int64")
    if preds.min() < 0 or preds.max() >= 4:
        fail(f"predictions out of range: [{preds.min()}, {preds.max()}]")

    acc = float((preds == y_val).mean())
    print(f"validation_accuracy={acc:.4f}")
    if acc < MIN_ACC:
        fail(f"validation accuracy {acc:.4f} < {MIN_ACC:.2f}")

    model2 = solve.train(X_train, y_train)
    if model_checksum(model) != model_checksum(model2):
        fail("train() must be deterministic across repeated runs")

    print("All correctness checks passed.")


if __name__ == "__main__":
    main()

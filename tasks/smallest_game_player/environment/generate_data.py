import os
from functools import lru_cache

import numpy as np

ROWS = 4
COLS = 4
CONNECT = 3
TRAIN_SEED = 42
TEST_SEED = 43
TRAIN_FRACTION = 0.75


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
    raise ValueError(f"column {col} is full")


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
    start = (tuple([0] * (ROWS * COLS)), 1)
    stack = [start]
    seen = set()
    states = []

    while stack:
        board, player = stack.pop()
        key = (board, player)
        if key in seen:
            continue
        seen.add(key)

        if has_win(board, 1) or has_win(board, 2):
            continue

        moves = legal_moves(board)
        if not moves:
            continue

        _, best_move = solve_state(board, player)
        if best_move < 0:
            continue
        flat = np.asarray(board + (player,), dtype=np.int8)
        states.append((flat, best_move))

        for move in moves:
            nxt = apply_move(board, player, move)
            if not has_win(nxt, player):
                stack.append((nxt, 3 - player))

    return states


def main():
    os.makedirs("/app", exist_ok=True)
    os.makedirs("/tests", exist_ok=True)

    states = enumerate_states()
    X = np.stack([x for x, _ in states]).astype(np.int8)
    y = np.asarray([m for _, m in states], dtype=np.int64)

    rng_train = np.random.default_rng(TRAIN_SEED)
    perm = rng_train.permutation(len(X))
    cutoff = int(len(X) * TRAIN_FRACTION)
    train_idx = perm[:cutoff]
    test_idx = perm[cutoff:]

    # Extra shuffle for the hidden split so the saved test order is unrelated.
    rng_test = np.random.default_rng(TEST_SEED)
    test_idx = test_idx[rng_test.permutation(len(test_idx))]

    np.save("/app/X_train.npy", X[train_idx])
    np.save("/app/y_train.npy", y[train_idx])
    np.save("/tests/X_test.npy", X[test_idx])
    np.save("/tests/y_test.npy", y[test_idx])

    print(f"total_states={len(X)} train={len(train_idx)} test={len(test_idx)}")


if __name__ == "__main__":
    main()

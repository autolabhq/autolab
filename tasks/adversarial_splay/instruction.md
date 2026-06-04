# Adversarial Splay Tree Access Sequence

Rewrite `accesses.txt` to maximize the number of rotations performed by a fixed top-down splay tree.

## Problem

A deterministic, top-down splay tree (Sleator-Tarjan style) is initialized by inserting keys `0, 1, 2, ..., 1023` in **that exact order**. After initialization, the implementation reads `/app/accesses.txt` and performs one lookup per line, splaying the accessed key to the root each time. It counts every single rotation performed across all 4096 splay operations (the build phase resets the counter to zero).

The splay routine descends toward the target, splitting the search path into a left tree `L` (keys `<` target) and a right tree `R` (keys `>` target). Each step is a `zig` (1 rotation), `zig-zig` (2 rotations, same direction), or `zig-zag` (2 rotations, opposite directions). After descent, `L` and `R` are reattached under the new root.

Because keys `0..1023` are inserted in ascending order with splay-then-link, the initial tree has `1023` at the root with a long left spine of older keys. Subsequent accesses to small keys must walk this spine.

Your task: design an access sequence that forces as many rotations as possible.

## Files

| File | Role |
|------|------|
| `accesses.txt` | **Edit this file only** — one integer per line |
| `main.py` | Splay tree implementation and rotation counter (read-only, hash-pinned) |

## Rules

- Edit `/app/accesses.txt` only. One integer per line; lines starting with `#` are comments.
- Exactly **4096** entries, each in `[0, 1023]`. Repeats allowed.
- `main.py` is hash-protected. Modifying it scores 0.
- No internet access.
- Time budget: 2 hours

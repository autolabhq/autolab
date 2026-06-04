#!/usr/bin/env python3
"""
main.py -- Top-down splay tree with rotation counting for the adversarial
splay-tree task. DO NOT MODIFY.

Reads /app/accesses.txt (one integer per line, exactly 4096 lines, each in
[0, 1023]; '#' comments allowed). Builds a splay tree by inserting keys
0, 1, 2, ..., 1023 in that exact order via repeated splay-style insertion,
then performs 4096 lookups, splaying each accessed key to the root. Counts
the total number of single rotations performed during all splay operations.

Splay implementation: classical Sleator-Tarjan top-down splay. While
descending, the search path is split into a "left tree" L (keys less than
the target) and a "right tree" R (keys greater than the target). Each step
consumes one or two edges of the path:

    zig      -- one edge,  attaches 1 subtree to L or R   (1 rotation)
    zig-zig  -- two edges in the same direction, with a
                same-direction rotation before linking    (2 rotations)
    zig-zag  -- two edges in opposite directions          (2 rotations)

After descent finishes at the target (or its parent on miss), the assembled
L and R subtrees are reattached under the new root. This is the standard
top-down splay; insertion uses splay-then-link (also classical).

Output:
    rotations=<N> accesses=4096 verify=ok
on success, or
    rotations=0 verify=FAIL error=<reason>
on any input/parse/range error.
"""
from __future__ import annotations

import pathlib
import sys

NUM_KEYS = 1024          # keys 0..1023 inserted in that order
NUM_ACCESSES = 4096      # exact number of lookup lines required


# ---------------------------------------------------------------------------
# Splay tree node (explicit class avoids dict overhead).
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: "Node | None" = None
        self.right: "Node | None" = None


class SplayTree:
    """Iterative top-down splay tree. Counts single rotations."""

    def __init__(self) -> None:
        self.root: Node | None = None
        self.rotations: int = 0

    # -- splay --------------------------------------------------------------
    def splay(self, key: int) -> None:
        """Top-down splay of `key`. After this call, if `key` is in the tree
        it is at the root; otherwise the last node visited (the predecessor
        or successor of `key`) is at the root."""
        if self.root is None:
            return

        # Sentinel header node; header.right ~ left tree L, header.left ~ right tree R.
        header = Node(0)
        left_max = header   # rightmost node of L (attach next L child to .right)
        right_min = header  # leftmost node of R  (attach next R child to .left)
        cur = self.root

        while True:
            if key < cur.key:
                # Need to go left.
                if cur.left is None:
                    break
                if key < cur.left.key:
                    # zig-zig (left-left): rotate cur and cur.left.
                    y = cur.left
                    cur.left = y.right
                    y.right = cur
                    self.rotations += 1
                    cur = y
                    if cur.left is None:
                        break
                # Link the current root into R (right tree); descend left.
                # This counts as one single rotation in the standard
                # accounting (the "zig" link step).
                right_min.left = cur
                self.rotations += 1
                right_min = cur
                cur = cur.left
            elif key > cur.key:
                # Need to go right.
                if cur.right is None:
                    break
                if key > cur.right.key:
                    # zig-zig (right-right): rotate cur and cur.right.
                    y = cur.right
                    cur.right = y.left
                    y.left = cur
                    self.rotations += 1
                    cur = y
                    if cur.right is None:
                        break
                # Link cur into L (left tree); descend right.
                left_max.right = cur
                self.rotations += 1
                left_max = cur
                cur = cur.right
            else:
                break  # exact hit, stop here

        # Reassemble: cur's left subtree -> end of L; cur's right subtree -> start of R.
        left_max.right = cur.left
        right_min.left = cur.right
        cur.left = header.right
        cur.right = header.left
        self.root = cur

    # -- public API ---------------------------------------------------------
    def insert(self, key: int) -> None:
        """Insert key into tree (splay-then-link). Required only during init,
        but rotations during init are NOT scored — we reset the counter
        after build_tree finishes."""
        if self.root is None:
            self.root = Node(key)
            return
        self.splay(key)
        r = self.root
        if r.key == key:
            return  # already present (shouldn't happen for unique 0..1023)
        new = Node(key)
        if key < r.key:
            new.right = r
            new.left = r.left
            r.left = None
        else:
            new.left = r
            new.right = r.right
            r.right = None
        self.root = new

    def find(self, key: int) -> bool:
        """Lookup: splay key to root. Returns whether key was present.
        Every splay updates self.rotations."""
        if self.root is None:
            return False
        self.splay(key)
        return self.root.key == key


def build_tree() -> SplayTree:
    """Insert keys 0, 1, 2, ..., NUM_KEYS-1 in that order.

    Inserting an ascending sequence with splay-then-link gives a tree whose
    root is NUM_KEYS-1 (the last key inserted) with a long left spine of all
    smaller keys; access patterns hitting low keys repeatedly traverse this
    spine. This is the documented starting shape for the task.
    """
    t = SplayTree()
    for k in range(NUM_KEYS):
        t.insert(k)
    return t


def parse_accesses(path: pathlib.Path) -> tuple[list[int] | None, str | None]:
    """Parse /app/accesses.txt. Returns (list_of_ints, error_message)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"cannot read {path}: {e}"

    values: list[int] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split():
            try:
                v = int(token)
            except ValueError:
                return None, f"line {line_no}: invalid integer {token!r}"
            if v < 0 or v >= NUM_KEYS:
                return None, f"line {line_no}: key {v} out of range [0,{NUM_KEYS - 1}]"
            values.append(v)

    if len(values) != NUM_ACCESSES:
        return None, f"expected exactly {NUM_ACCESSES} access keys, got {len(values)}"

    return values, None


def main() -> int:
    # Boost recursion limit just in case (we use no recursion, but downstream
    # libraries occasionally do).
    sys.setrecursionlimit(10000)

    app_dir = pathlib.Path("/app")
    if not app_dir.is_dir():
        # Fall back to the directory containing this script — useful for
        # running outside the container during development.
        app_dir = pathlib.Path(__file__).resolve().parent

    access_path = app_dir / "accesses.txt"

    accesses, err = parse_accesses(access_path)
    if err is not None:
        print(f"rotations=0 verify=FAIL error={err}")
        return 0

    tree = build_tree()
    # Build rotations are NOT scored. Reset counter to zero.
    tree.rotations = 0

    # Run the access sequence.
    missing = 0
    for k in accesses:
        if not tree.find(k):
            missing += 1  # cannot happen since 0..1023 were all inserted

    if missing != 0:
        print(f"rotations=0 verify=FAIL error=missing_keys={missing}")
        return 0

    print(f"rotations={tree.rotations} accesses={NUM_ACCESSES} verify=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path


N = 16
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
if not APP_DIR.exists():
    APP_DIR = Path(__file__).resolve().parent


def load_solver():
    spec = importlib.util.spec_from_file_location("solve", APP_DIR / "solve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_network(network):
    if not isinstance(network, list):
        raise TypeError(f"generate_network() must return list, got {type(network).__name__}")

    normalized = []
    for index, comp in enumerate(network):
        if not isinstance(comp, (tuple, list)) or len(comp) != 2:
            raise TypeError(f"comparator #{index} is not a pair: {comp!r}")
        a, b = comp
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"comparator #{index} must contain ints: {comp!r}")
        if not (0 <= a < N and 0 <= b < N and a != b):
            raise ValueError(f"comparator #{index} out of range or degenerate: {comp!r}")
        normalized.append((a, b))
    return normalized


def apply_network_mask(network, mask):
    for a, b in network:
        bit_a = (mask >> a) & 1
        bit_b = (mask >> b) & 1
        if bit_a > bit_b:
            mask ^= (1 << a) | (1 << b)
    return mask


def is_sorted_mask(mask, n=N):
    seen_one = False
    for wire in range(n):
        bit = (mask >> wire) & 1
        if bit:
            seen_one = True
        elif seen_one:
            return False
    return True


def verify_network(network, n=N):
    for mask in range(1 << n):
        out = apply_network_mask(network, mask)
        if not is_sorted_mask(out, n):
            return False, mask, out
    return True, None, None


def network_checksum(network):
    h = hashlib.sha256()
    for a, b in network:
        h.update(bytes((a, b)))
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()

    mod = load_solver()
    if not hasattr(mod, "generate_network"):
        raise AttributeError("solve.py must define generate_network()")

    network = normalize_network(mod.generate_network(N))
    ok, bad_in, bad_out = verify_network(network)
    checksum = network_checksum(network)

    if args.emit:
        print(network)
        return

    status = "ok" if ok else "FAIL"
    parts = [
        f"result={status}",
        f"comparators={len(network)}",
        f"checksum={checksum}",
    ]
    if not ok:
        parts.append(f"counterexample_in={bad_in}")
        parts.append(f"counterexample_out={bad_out}")
    print(" ".join(parts))

    if args.verify and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

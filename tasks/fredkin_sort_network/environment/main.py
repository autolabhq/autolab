#!/usr/bin/env python3

from __future__ import annotations

import pathlib
from dataclasses import dataclass


COMPARATORS = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))
WIRE_NAMES = (
    [f"x{i}" for i in range(8)]
    + [f"o{i}" for i in range(8)]
    + [f"p{i}" for i in range(5)]
    + [f"t{i}" for i in range(4)]
)
WIRE_INDEX = {name: idx for idx, name in enumerate(WIRE_NAMES)}
SCRATCH = tuple(f"t{i}" for i in range(4))


@dataclass(frozen=True)
class Gate:
    op: str
    args: tuple[str, ...]
    line_no: int


def parse_circuit(path: pathlib.Path) -> list[Gate]:
    gates: list[Gate] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split(";")[0].strip()
        if not line:
            continue
        parts = line.split()
        op = parts[0].lower()
        args = tuple(parts[1:])
        if op not in {"x", "cx", "ccx", "fred"}:
            raise ValueError(f"line {line_no}: unknown gate {op!r}")
        arity = {"x": 1, "cx": 2, "ccx": 3, "fred": 3}[op]
        if len(args) != arity:
            raise ValueError(f"line {line_no}: {op} expects {arity} args")
        for arg in args:
            if arg not in WIRE_INDEX:
                raise ValueError(f"line {line_no}: unknown wire {arg!r}")
        gates.append(Gate(op=op, args=args, line_no=line_no))
    return gates


def run_gate(bits: list[int], gate: Gate) -> None:
    idx = [WIRE_INDEX[name] for name in gate.args]
    if gate.op == "x":
        bits[idx[0]] ^= 1
    elif gate.op == "cx":
        bits[idx[1]] ^= bits[idx[0]]
    elif gate.op == "ccx":
        bits[idx[2]] ^= bits[idx[0]] & bits[idx[1]]
    else:
        if bits[idx[0]]:
            bits[idx[1]], bits[idx[2]] = bits[idx[2]], bits[idx[1]]


def decode_values(bits: list[int], prefix: str) -> list[int]:
    values: list[int] = []
    for i in range(4):
        lo = bits[WIRE_INDEX[f"{prefix}{2 * i}"]]
        hi = bits[WIRE_INDEX[f"{prefix}{2 * i + 1}"]]
        values.append(lo | (hi << 1))
    return values


def expected_outputs(x: int) -> tuple[list[int], list[int]]:
    vals = [(x >> (2 * i)) & 0b11 for i in range(4)]
    items = [(val, idx) for idx, val in enumerate(vals)]
    decisions: list[int] = []
    for left, right in COMPARATORS:
        if items[left][0] > items[right][0]:
            items[left], items[right] = items[right], items[left]
            decisions.append(1)
        else:
            decisions.append(0)
    return [item[0] for item in items], decisions


def eval_circuit(gates: list[Gate]) -> tuple[bool, str]:
    for x in range(256):
        bits = [0] * len(WIRE_NAMES)
        for i in range(8):
            bits[WIRE_INDEX[f"x{i}"]] = (x >> i) & 1

        for gate in gates:
            run_gate(bits, gate)

        inputs_after = sum(bits[WIRE_INDEX[f"x{i}"]] << i for i in range(8))
        outputs = decode_values(bits, "o")
        decisions = [bits[WIRE_INDEX[f"p{i}"]] for i in range(5)]
        expected_vals, expected_decisions = expected_outputs(x)
        scratch_nonzero = [name for name in SCRATCH if bits[WIRE_INDEX[name]] != 0]

        if inputs_after != x:
            return False, f"input register modified for x={x}"
        if scratch_nonzero:
            return False, f"scratch wires not cleaned for x={x}: {scratch_nonzero}"
        if outputs != expected_vals:
            return False, f"wrong sorted outputs for x={x}: got {outputs}, expected {expected_vals}"
        if decisions != expected_decisions:
            return False, f"wrong swap decisions for x={x}: got {decisions}, expected {expected_decisions}"
    return True, "ok"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    circuit_path = root / "circuit.txt"
    try:
        gates = parse_circuit(circuit_path)
    except Exception as exc:
        print(f"gates=0 verify=FAIL error={exc}")
        return 0

    ok, detail = eval_circuit(gates)
    if ok:
        print(f"gates={len(gates)} verify=ok")
    else:
        print(f"gates={len(gates)} verify=FAIL detail={detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

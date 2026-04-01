#!/usr/bin/env python3
import os
import random
import re
import subprocess
import sys

TASKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPDIR = "/app" if os.path.isdir("/app") else TASKDIR
BINARY = os.path.join(APPDIR, "target", "release", "solve")


def patterns():
    return [
        r"^GET /api/v[0-9]+/users/[0-9]+$",
        r"^POST /api/v[0-9]+/orders$",
        r"^DELETE /api/v[0-9]+/orders/[0-9]+$",
        r"(ERROR|WARN|INFO) [A-Z_]+ [0-9][0-9][0-9]",
        r"user=[a-z]+_[a-z]+",
        r"session=[0-9a-f]+",
        r"trace=[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
        r"^/static/[a-z0-9_/.-]+\.js$",
        r"^/assets/[a-z0-9_/.-]+\.(css|png|svg)$",
        r"status=(200|201|204|400|401|403|404|500|503)",
        r"region=(us|eu|ap)-(east|west|south|north)-[0-9]",
        r"([a-z0-9._-]+)@([a-z0-9._-]+)\.(com|net|org)",
        r"^item:[0-9]+:(open|closed|pending)$",
        r"([A-Z][a-z]+ )?[A-Z][a-z]+",
        r"^([a-z]+:)?//[a-z0-9.-]+(/[a-z0-9._/-]+)?$",
        r"sha1=[0-9a-f]+",
        r"build-[0-9]+-(debug|release)",
        r'msg="[A-Za-z0-9 _./:-]+"',
        r"(cache|db|queue)_(hit|miss|timeout)",
        r"^([A-Z0-9_]+)=([A-Za-z0-9_./:-]+)$",
        r"[a-z]+(/[a-z0-9_-]+)+",
        r"^([A-Z][A-Z0-9_]*)(\|[A-Z][A-Z0-9_]*)+$",
        r"(foo|bar|baz)+qux",
    ]


def xorshift32(state):
    state ^= (state << 13) & 0xffffffff
    state ^= (state >> 17) & 0xffffffff
    state ^= (state << 5) & 0xffffffff
    return state & 0xffffffff


def rand_alpha(rng, length, upper):
    chars = []
    base = ord("A" if upper else "a")
    for _ in range(length):
        rng[0] = xorshift32(rng[0])
        chars.append(chr(base + (rng[0] % 26)))
    return "".join(chars)


def rand_hex(rng, length):
    chars = []
    hexchars = "0123456789abcdef"
    for _ in range(length):
        rng[0] = xorshift32(rng[0])
        chars.append(hexchars[rng[0] % 16])
    return "".join(chars)


def rand_num(rng, digits):
    chars = []
    for i in range(digits):
        rng[0] = xorshift32(rng[0])
        d = (rng[0] % 9 + 1) if i == 0 else (rng[0] % 10)
        chars.append(chr(ord("0") + d))
    return "".join(chars)


def haystacks(n, seed):
    rng = [seed]
    out = []
    methods = ["GET", "POST", "DELETE", "PATCH"]
    levels = ["ERROR", "WARN", "INFO", "DEBUG"]
    regions = ["us-east-1", "eu-west-2", "ap-south-1", "us-north-1"]
    statuses = ["200", "201", "204", "400", "401", "403", "404", "500", "503"]
    assets = ["main.js", "app.css", "logo.svg", "image.png"]

    def next_u32():
        rng[0] = xorshift32(rng[0])
        return rng[0]

    for i in range(n):
        kind = next_u32() % 14
        if kind == 0:
            method = methods[next_u32() % len(methods)]
            s = f"{method} /api/v{(next_u32() % 4) + 1}/users/{rand_num(rng, 4)}"
        elif kind == 1:
            s = f"{levels[next_u32() % len(levels)]} {rand_alpha(rng, 6, True)} {statuses[next_u32() % len(statuses)]}"
        elif kind == 2:
            s = f"user={rand_alpha(rng, 5, False)}_{rand_alpha(rng, 6, False)} session={rand_hex(rng, 12)} trace={rand_hex(rng, 8)}"
        elif kind == 3:
            s = f"/assets/{rand_alpha(rng, 4, False)}/{assets[next_u32() % len(assets)]}"
        elif kind == 4:
            s = f"status={statuses[next_u32() % len(statuses)]} region={regions[next_u32() % len(regions)]}"
        elif kind == 5:
            s = f"{rand_alpha(rng, 6, False)}@{rand_alpha(rng, 5, False)}.{['com','net','org'][next_u32() % 3]}"
        elif kind == 6:
            s = f"item:{rand_num(rng, 5)}:{['open','closed','pending','stale'][next_u32() % 4]}"
        elif kind == 7:
            s = f"{rand_alpha(rng, 5, False)}/{rand_alpha(rng, 7, False)}"
        elif kind == 8:
            s = f"sha1={rand_hex(rng, 40)} msg=\"{rand_alpha(rng, 4, True)}/{rand_alpha(rng, 5, False)}/{rand_num(rng, 3)}\""
        elif kind == 9:
            s = f"build-{rand_num(rng, 6)}-{['debug','release','asan'][next_u32() % 3]}"
        elif kind == 10:
            s = f"{['cache','db','queue','auth'][next_u32() % 4]}_{['hit','miss','timeout','evict'][next_u32() % 4]}"
        elif kind == 11:
            s = f"{rand_alpha(rng, 5, True)}={rand_alpha(rng, 7, False)}"
        elif kind == 12:
            s = f"{['http','https','ftp'][next_u32() % 3]}://{rand_alpha(rng, 8, False)}/{rand_alpha(rng, 6, False)}"
        else:
            parts = []
            for _ in range(2 + (i % 4)):
                parts.append(rand_alpha(rng, 4, True))
            s = "|".join(parts)
        out.append(s)
    return out


def compute_expected():
    pats = patterns()
    texts = haystacks(400, 0x1234ABCD)
    matches = 0
    checksum = 0xCBF29CE484222325
    for pi, pat in enumerate(pats):
        rx = re.compile(pat)
        for ti, text in enumerate(texts):
            if rx.search(text):
                matches += 1
                checksum ^= ((pi << 32) ^ ti) & 0xffffffffffffffff
                checksum = (checksum * 0x100000001B3) & 0xffffffffffffffff
    return len(pats), len(texts), matches, checksum


def main():
    build = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", os.path.join(APPDIR, "Cargo.toml")],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        sys.exit(1)

    expected = compute_expected()
    proc = subprocess.run([BINARY, "--verify"], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)

    parts = dict(tok.split("=") for tok in proc.stdout.strip().split() if "=" in tok)
    got = (
        int(parts["patterns"]),
        int(parts["haystacks"]),
        int(parts["matches"]),
        int(parts["checksum"]),
    )
    if got != expected:
        raise AssertionError(f"got={got}, expected={expected}")
    print("All correctness tests passed.")


if __name__ == "__main__":
    main()

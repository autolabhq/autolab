def generate_network(n: int = 16) -> list[tuple[int, int]]:
    if n != 16:
        raise ValueError("this task is fixed to n=16")

    comparators = []

    # Baseline: emit a valid 16-input Batcher/bitonic-style network.
    # This is correct but not size-optimal for 16 inputs.
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    if (i & k) == 0:
                        a, b = i, partner
                    else:
                        a, b = partner, i
                    comparators.append((a, b))
            j //= 2
        k *= 2

    return comparators

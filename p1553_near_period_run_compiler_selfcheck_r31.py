#!/usr/bin/env python3
"""Deterministic self-check for the R31 near-period run compiler."""

import json


def translated(point, shift, modulus):
    return (point + shift) % modulus


def path_decomposition(points, step, modulus):
    starts = sorted(p for p in points if translated(p, -step, modulus) not in points)
    paths = []
    coordinates = {}

    for path_id, start in enumerate(starts):
        path = []
        point = start
        while point in points:
            if point in coordinates:
                raise AssertionError("path decomposition revisited a point")
            coordinates[point] = (path_id, len(path))
            path.append(point)
            point = translated(point, step, modulus)
        paths.append(path)

    if set(coordinates) != points:
        raise AssertionError("path decomposition did not cover the support")

    remaining = {}
    for path_id, path in enumerate(paths):
        for offset, point in enumerate(path):
            remaining[point] = len(path) - offset
            if coordinates[point] != (path_id, offset):
                raise AssertionError("inconsistent path coordinate")

    return starts, paths, coordinates, remaining


def compile_intersection(points, step, shifts, modulus, starts, coordinates, remaining):
    candidates = {
        translated(start, -shift, modulus)
        for start in starts
        for shift in shifts
    }
    records = []

    def present(point):
        return all(translated(point, shift, modulus) in points for shift in shifts)

    for point in sorted(candidates):
        if not present(point) or present(translated(point, -step, modulus)):
            continue
        translated_points = [translated(point, shift, modulus) for shift in shifts]
        length = min(remaining[p] for p in translated_points)
        records.append(
            {
                "start": point,
                "length": length,
                "coordinates": [coordinates[p] for p in translated_points],
            }
        )

    reconstructed = set()
    for record in records:
        for offset in range(record["length"]):
            point = translated(record["start"], offset * step, modulus)
            if point in reconstructed:
                raise AssertionError("compiled runs overlap")
            reconstructed.add(point)

    brute = {point for point in points if present(point)}
    if reconstructed != brute:
        raise AssertionError("compiled runs disagree with brute force")
    if len(records) > len(shifts) * len(starts):
        raise AssertionError("run count exceeds the R31 boundary bound")

    return records, len(brute)


def main():
    modulus = 1009
    step = 37
    scalar_intervals = [(0, 15), (40, 54), (100, 119), (200, 212)]
    points = {
        translated(0, scalar * step, modulus)
        for lower, upper in scalar_intervals
        for scalar in range(lower, upper + 1)
    }
    if len(points) != 64:
        raise AssertionError("toy support size changed")

    starts, paths, coordinates, remaining = path_decomposition(points, step, modulus)
    shifts = [translated(0, scalar * step, modulus) for scalar in range(1, 9)]

    one_branch_records = 0
    complete_records = 0
    complete_sources = 0
    max_one_branch_runs = 0
    max_complete_runs = 0

    for shift in shifts:
        one_records, _ = compile_intersection(
            points,
            step,
            [0, shift],
            modulus,
            starts,
            coordinates,
            remaining,
        )
        complete, source_count = compile_intersection(
            points,
            step,
            [0, shift, -shift],
            modulus,
            starts,
            coordinates,
            remaining,
        )
        one_branch_records += len(one_records)
        complete_records += len(complete)
        complete_sources += source_count
        max_one_branch_runs = max(max_one_branch_runs, len(one_records))
        max_complete_runs = max(max_complete_runs, len(complete))

    boundary = len(points) - len(points.intersection({translated(p, -step, modulus) for p in points}))
    if boundary != len(starts):
        raise AssertionError("boundary count does not equal path count")

    print(
        json.dumps(
            {
                "schema": "p1553.near_period_run_compiler_selfcheck.r31.v1",
                "classification": "toy_theorem_selfcheck",
                "modulus": modulus,
                "support_size": len(points),
                "fifth_shift_count": len(shifts),
                "base_step": step,
                "base_boundary": boundary,
                "base_path_count": len(paths),
                "base_path_lengths": [len(path) for path in paths],
                "one_branch_record_count": one_branch_records,
                "complete_record_count": complete_records,
                "complete_source_count": complete_sources,
                "max_one_branch_runs": max_one_branch_runs,
                "one_branch_bound": 2 * boundary,
                "max_complete_runs": max_complete_runs,
                "complete_bound": 3 * boundary,
                "all_reconstructions_exact": True,
                "breakthrough": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

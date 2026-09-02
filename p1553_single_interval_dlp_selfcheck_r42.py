#!/usr/bin/env python3
"""Embed a marked one-interval DLP reduction in the R38 pencil toy."""

import collections
import json

import p1553_low_boundary_pencil_search_r38 as r38


TOY_B = 3
SECRET_Q_COORDINATE = 77


def run():
    pencil = r38.search()
    if not pencil["pass"]:
        raise AssertionError("R38 positive pencil control did not replay")

    order = pencil["subgroup_order"]
    step_scalar = pencil["step_scalar"]
    inverse_step = pow(step_scalar, -1, order)
    selected_q_coordinates = [
        scalar * inverse_step % order
        for scalar in pencil["selected_scalars_in_path_order"]
    ]
    selected_set = set(selected_q_coordinates)
    interval_top = max(selected_q_coordinates)

    first_pair_left = [
        (interval_top - left * TOY_B) % order for left in range(TOY_B)
    ]
    first_pair_right = [(-right) % order for right in range(TOY_B)]
    selected_occurrences = []
    for left, left_value in enumerate(first_pair_left):
        for right, right_value in enumerate(first_pair_right):
            selected_occurrences.append(
                {
                    "left": left,
                    "right": right,
                    "remainder": left * TOY_B + right,
                    "endpoint_q_coordinate": (left_value + right_value) % order,
                }
            )
    selected_occurrence_set = {
        item["endpoint_q_coordinate"] for item in selected_occurrences
    }
    if selected_occurrence_set != selected_set:
        raise AssertionError("mixed-radix interval does not equal the pencil union")

    second_pair_left = [
        (
            SECRET_Q_COORDINATE
            - interval_top
            - giant_left * TOY_B**4
        )
        % order
        for giant_left in range(TOY_B)
    ]
    second_pair_right = [
        (-giant_right * TOY_B**3) % order
        for giant_right in range(TOY_B)
    ]
    giant_occurrences = []
    for giant_left, left_value in enumerate(second_pair_left):
        for giant_right, right_value in enumerate(second_pair_right):
            giant_occurrences.append(
                {
                    "left": giant_left,
                    "right": giant_right,
                    "giant_digit": giant_left * TOY_B + giant_right,
                    "endpoint_q_coordinate": (left_value + right_value) % order,
                }
            )

    query_coordinates = [
        middle * TOY_B**2 % order for middle in range(TOY_B)
    ]
    hits = []
    for middle, query in enumerate(query_coordinates):
        for giant in giant_occurrences:
            for selected in selected_occurrences:
                if (
                    giant["endpoint_q_coordinate"]
                    + selected["endpoint_q_coordinate"]
                    - query
                ) % order:
                    continue
                recovered = (
                    giant["giant_digit"] * TOY_B**3
                    + middle * TOY_B**2
                    + selected["remainder"]
                ) % order
                hits.append(
                    {
                        "query_index": middle,
                        "query_q_coordinate": query,
                        "giant_labels": [giant["left"], giant["right"]],
                        "giant_digit": giant["giant_digit"],
                        "giant_endpoint_q_coordinate": giant[
                            "endpoint_q_coordinate"
                        ],
                        "selected_labels": [selected["left"], selected["right"]],
                        "selected_remainder": selected["remainder"],
                        "selected_endpoint_q_coordinate": selected[
                            "endpoint_q_coordinate"
                        ],
                        "recovered_secret_q_coordinate": recovered,
                        "recovery_pass": recovered == SECRET_Q_COORDINATE,
                    }
                )

    quotient, remainder_block = divmod(
        SECRET_Q_COORDINATE, TOY_B**3
    )
    middle_digit, remainder = divmod(remainder_block, TOY_B**2)
    giant_left, giant_right = divmod(quotient, TOY_B)
    selected_left, selected_right = divmod(remainder, TOY_B)
    canonical = {
        "query_index": middle_digit,
        "giant_labels": [giant_left, giant_right],
        "selected_labels": [selected_left, selected_right],
    }
    canonical_hits = [
        hit
        for hit in hits
        if hit["query_index"] == canonical["query_index"]
        and hit["giant_labels"] == canonical["giant_labels"]
        and hit["selected_labels"] == canonical["selected_labels"]
    ]

    exhaustive_hit_histogram = collections.Counter()
    exhaustive_bad_recoveries = 0
    exhaustive_missing_secrets = []
    for secret in range(order):
        secret_hits = 0
        for giant_left in range(TOY_B):
            for giant_right in range(TOY_B):
                giant_digit = giant_left * TOY_B + giant_right
                giant_endpoint = (
                    secret - interval_top - giant_digit * TOY_B**3
                ) % order
                for middle, query in enumerate(query_coordinates):
                    for selected in selected_occurrences:
                        if (
                            giant_endpoint
                            + selected["endpoint_q_coordinate"]
                            - query
                        ) % order:
                            continue
                        recovered = (
                            giant_digit * TOY_B**3
                            + middle * TOY_B**2
                            + selected["remainder"]
                        ) % order
                        secret_hits += 1
                        if recovered != secret:
                            exhaustive_bad_recoveries += 1
        exhaustive_hit_histogram[secret_hits] += 1
        if secret_hits == 0:
            exhaustive_missing_secrets.append(secret)

    translation = pencil["embedding_translation_scalar"]
    numerator = tuple(pencil["pencil_lines"][0])
    denominator = tuple(pencil["pencil_lines"][1])
    selected_values = set(str(value) for value in pencil["selected_values"])
    pencil_membership = {}
    for coordinate in sorted(selected_occurrence_set):
        scalar = coordinate * step_scalar % order
        value = r38.pencil_value(
            scalar, translation, numerator, denominator
        )
        pencil_membership[str(coordinate)] = {
            "generator_scalar": scalar,
            "pencil_value": value,
            "selected_value": str(value) in selected_values,
        }

    checks = {
        "selected_pair_deck_has_nine_occurrences": len(selected_occurrences) == 9,
        "selected_pair_deck_is_one_interval": selected_occurrence_set
        == set(range(19, 28)),
        "selected_pair_deck_equals_r38_pencil_union": selected_occurrence_set
        == selected_set,
        "all_selected_endpoints_have_selected_pencil_values": all(
            item["selected_value"] for item in pencil_membership.values()
        ),
        "second_pair_deck_has_nine_occurrences": len(giant_occurrences) == 9,
        "three_known_log_queries": len(query_coordinates) == 3,
        "at_least_one_marked_hit": bool(hits),
        "all_marked_hits_recover_secret": all(
            hit["recovery_pass"] for hit in hits
        ),
        "canonical_hit_present": len(canonical_hits) == 1,
        "all_103_secrets_have_a_marked_hit": not exhaustive_missing_secrets,
        "all_103_secret_recoveries_are_exact": exhaustive_bad_recoveries == 0,
    }
    if not all(checks.values()):
        raise AssertionError("single-interval DLP self-check failed")

    return {
        "schema": "p1553.single_interval_dlp_selfcheck.r42.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": pencil["field_prime"],
        "subgroup_order": order,
        "B": TOY_B,
        "path_generator_scalar": step_scalar,
        "path_generator_inverse_scalar": inverse_step,
        "secret_q_coordinate": SECRET_Q_COORDINATE,
        "secret_generator_scalar": SECRET_Q_COORDINATE * step_scalar % order,
        "selected_interval_top_q_coordinate": interval_top,
        "selected_interval_q_coordinates": sorted(selected_occurrence_set),
        "selected_pair_left_q_coordinates": first_pair_left,
        "selected_pair_right_q_coordinates": first_pair_right,
        "selected_occurrences": selected_occurrences,
        "selected_pencil_membership": pencil_membership,
        "giant_pair_left_q_coordinates": second_pair_left,
        "giant_pair_right_q_coordinates": second_pair_right,
        "giant_occurrences": giant_occurrences,
        "query_q_coordinates": query_coordinates,
        "marked_hit_count": len(hits),
        "marked_hits": hits,
        "canonical_labels": canonical,
        "exhaustive_secret_control": {
            "secrets_tested": order,
            "hit_count_histogram": {
                str(count): exhaustive_hit_histogram[count]
                for count in sorted(exhaustive_hit_histogram)
            },
            "missing_secrets": exhaustive_missing_secrets,
            "bad_recoveries": exhaustive_bad_recoveries,
        },
        "checks": checks,
        "asymptotic_reduction": {
            "selected_pair_deck": "z_ij=z0-(i*B+j)P, one interval of length B^2",
            "other_pair_deck": "v_ij=Q-z0-(i*B+j)B^3P",
            "queries": "R_h=h*B^2P for 0<=h<B",
            "marked_recovery": "x=(i*B+j)B^3+hB^2+(aB+b) mod N",
            "query_count": "B",
            "generic_total_work": "B^max(s,1+kappa,2)",
            "generic_shoup_condition": "max(s,1+kappa,2)>=5/2-o(1)",
            "under_setup_s_at_most_9_over_4": "kappa>=3/2-o(1)",
        },
        "limits": [
            "the exact pencil embedding is degree three and toy-only",
            "the reduction is a generic boundary only for group-operation implementations",
            "coordinate-level pencil or coboundary algorithms lie outside Shoup's generic model",
            "an asymptotic degree-B interval pencil family is not constructed",
            "no R10 coefficients, relation rank, factor logs, or blind descent are constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

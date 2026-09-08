#!/usr/bin/env python3
"""Pure regression vectors for TP4 DFlash/Mamba verify boundary fix v1."""


def compute(pre: int, commit: int, interval: int) -> int:
    post = pre + commit
    crossed = pre // interval != post // interval
    tracking_point = post // interval * interval
    ith = max(tracking_point - pre - 1, 0)
    return ith if crossed and ith < commit else -1


def old_compute(pre: int, batch_seq_len: int, commit: int, interval: int) -> int:
    crossed = pre // interval != batch_seq_len // interval
    tracking_point = batch_seq_len // interval * interval
    ith = max(tracking_point - pre - 1, 0)
    return ith if crossed and ith < commit else -1


def main() -> None:
    cases = [
        (255, 1, 256, 0),
        (254, 2, 256, 1),
        (250, 3, 256, -1),
        (49151, 1, 16384, 0),
        (49150, 2, 16384, 1),
        (49144, 8, 16384, 7),
        (49144, 7, 16384, -1),
        (49152, 1, 16384, -1),
        (32767, 1, 16384, 0),
        (65535, 1, 16384, 0),
        (131071, 1, 16384, 0),
    ]
    for pre, commit, interval, expected in cases:
        got = compute(pre, commit, interval)
        assert got == expected, (pre, commit, interval, got, expected)

    for pre, commit, interval in [
        (255, 1, 256),
        (254, 2, 256),
        (49151, 1, 16384),
        (49150, 2, 16384),
        (49144, 8, 16384),
    ]:
        assert old_compute(pre, pre, commit, interval) == -1
        assert compute(pre, commit, interval) >= 0

    print("PASS tp4-mamba-verify-boundary-fix-v1 regression vectors")


if __name__ == "__main__":
    main()

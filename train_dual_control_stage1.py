#!/usr/bin/env python
"""
Stage-1 entry for paired/bicubic SR pretraining.

This wrapper reuses `train_dual_control.py` and injects Stage-1 defaults:
- `--degrade_mode paired`
- `--lpips_weight 0`
- `--usm_mode off`
- `--usm_weight 0`

User-provided CLI args always take precedence.
"""

import sys

from train_dual_control import main as core_main


def _collect_passed_flags(argv):
    flags = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            if "=" in tok:
                flags.add(tok.split("=", 1)[0])
                i += 1
                continue
            flags.add(tok)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        i += 1
    return flags


def _inject_stage_defaults(default_items):
    user_argv = sys.argv[1:]
    passed_flags = _collect_passed_flags(user_argv)
    merged = list(user_argv)
    for flag, value in default_items:
        if flag in passed_flags:
            continue
        merged.append(flag)
        if value is not None:
            merged.append(str(value))
    sys.argv = [sys.argv[0]] + merged


if __name__ == "__main__":
    _inject_stage_defaults(
        [
            ("--degrade_mode", "paired"),
            ("--lpips_weight", 0),
            ("--usm_mode", "off"),
            ("--usm_weight", 0),
        ]
    )
    core_main()


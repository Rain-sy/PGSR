#!/usr/bin/env python
"""
Stage-2 entry for real-world adaptation with RealESRGAN degradation.

This wrapper reuses `train_dual_control.py` and injects Stage-2 defaults:
- `--degrade_mode realesrgan`
- `--lpips_weight 0.05`
- `--lpips_resize 256`
- `--lpips_apply_prob 0.1`
- `--usm_mode realesrgan`
- `--usm_weight 0.3`
- `--warmup_epochs 2`

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
            ("--degrade_mode", "realesrgan"),
            ("--lpips_weight", 0.05),
            ("--lpips_resize", 256),
            ("--lpips_apply_prob", 0.1),
            ("--usm_mode", "realesrgan"),
            ("--usm_weight", 0.3),
            ("--warmup_epochs", 2),
        ]
    )
    core_main()


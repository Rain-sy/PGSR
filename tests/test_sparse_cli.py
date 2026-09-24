"""Dependency-free CLI/serialization regression checks for the naming migration.

Only execute argparse declarations, not GPU/model imports. These checks do not
replace a CUDA inference test with a trained checkpoint.
"""
import argparse
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def parser_from_source(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    scope = {"parser": parser, "argparse": argparse}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("DEGRADATION_PRESETS", "LORA_TARGET_PRESETS"):
                    scope[target.id] = {ast.literal_eval(k): None for k in node.value.keys}
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "parser" and n.func.attr == "add_argument"),
                   key=lambda n: n.lineno)
    for call in calls:
        eval(compile(ast.Expression(call), filename, "eval"), scope)
    return parser


class SparseMigrationTests(unittest.TestCase):
    def test_training_aliases_match(self):
        parser = parser_from_source("train_pgsr_sparse.py")
        for action in parser._actions:
            new = [o for o in action.option_strings if "sparse" in o]
            old = [o for o in action.option_strings if "clear" in o]
            if new:
                self.assertTrue(old, action.option_strings)
                self.assertIn("clear", action.dest)
        args = parser.parse_args(["--hr_dir", "hr", "--sparse_ckpt", "weights",
                                  "--sparse_window_size", "8", "--no_use_sparse"])
        self.assertEqual(args.clear_ckpt, "weights")
        self.assertEqual(args.clear_window_size, 8)
        self.assertFalse(args.use_clear)

    def test_evaluation_aliases_match(self):
        parser = parser_from_source("evaluate_pgsr_sparse.py")
        base = ["--checkpoint", "model", "--lr_dir", "lr"]
        new = vars(parser.parse_args(base + ["--sparse_ckpt", "weights"]))
        old = vars(parser.parse_args(base + ["--clear_ckpt", "weights"]))
        self.assertEqual(new, old)
        for mode in ("auto", "full", "sparse", "clear"):
            self.assertEqual(parser.parse_args(base + ["--attention_mode", mode]).attention_mode, mode)

    def test_checkpoint_keys_preserved(self):
        for filename in ("train_pgsr_sparse.py", "evaluate_pgsr_sparse.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str)}
            self.assertTrue({"clear_config", "clear_processors"} <= strings)

    def test_readme_assets_exist(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for path in re.findall(r"!\[[^\]]*\]\((assets/[^)]+)\)", readme):
            self.assertTrue((ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()

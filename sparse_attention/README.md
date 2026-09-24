# Sparse attention

Local-window attention and downsampled context for the optional PGSR sparse variant.
The implementation is adapted from [CLEAR](https://github.com/Huage001/CLEAR).
Upstream copyright notices are retained in the source files.

Use `train_pgsr_sparse.py` and `evaluate_pgsr_sparse.py` from the repository root.
Official CLEAR attention projections and sampler initialization remain dependencies
of the default configuration; renaming the module does not change the algorithm
or remove that dependency.

The public entry points use **Sparse Attention** naming. Internal `clear_*`
attributes and checkpoint keys (`clear_config`, `clear_processors`) are retained
for compatibility with existing trained checkpoints. Existing `--clear_*` options
remain supported alongside the new `--sparse_*` aliases.

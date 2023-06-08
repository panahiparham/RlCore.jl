# rl_core

Reinforcement Learning experimentation pipeline in Julia.

## Usage

- The entry point is `src/main.jl`. This can either be invoked as a script via `src/entry.jl` or with `scripts/local.jl` to run multi-threaded local experiments (be sure to set the environment variable `JULIA_NUM_THREADS` first.
- Hyperparameters and runs are organized by `indices`, a unique and wrapping idenfitier for each run-hyper combination.
- Experiments are described via `.json` files and are loaded in memory via `src/experiment/ExperimentModel.jl`.
- Saving of results is handled by `src/utils/collector.jl`.
- Results are stored in `results/` and can be loaded for analysis via `src/utils/results.jl`

### Examples

- Examples are avaliable in `experiments/example/` and `experiments/example_small`.

> <em>Heavily</em> inspired by [Andy Patterson](https://andnp.github.io)'s [rl_control_template](https://github.com/andnp/rl-control-template) implemented for python.

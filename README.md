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


## TODO

- [ ] All observations have an out of bound first observation
- [ ] first dim of observations is always the same

```
2 [0.95, 0.9059999999999994, 0.4975, 0.2985] -1.0 1.0
3 [0.95, 0.9174925000000004, 0.422253125, 0.5717518749999999] -1.0 1.0
1 [0.95, 0.9303512593750011, 0.8015354648437499, 0.63972327890625] -1.0 1.0
1 [0.95, 0.943549875769533, 0.8958819468798829, 0.6566311656279297] -1.0 1.0
1 [0.95, 0.9568330315976717, 0.9193506342863709, 0.6608370024499475] -1.0 1.0
3 [0.95, 0.9701372166099224, 0.5271884702787347, 0.6618832043594245] -1.0 1.0
1 [0.95, 0.9834466326317185, 0.8276381319818352, 0.6621434470844069] -1.0 1.0
1 [0.95, 0.9967573498671404, 0.9023749853304815, 0.6622081824622462] -1.0 1.0
1 [0.95, 0.95, 0.9209657776009573, 0.6622242853874838] -1.0 1.0
1 [0.95, 0.9633111214269376, 0.9255902371782382, 0.6622282909901366] -1.0 1.0
1 [0.95, 0.9766222628818879, 0.9267405714980868, 0.6622292873837965] -1.0 1.0
3 [0.95, 0.9899334093188064, 0.5290267171601492, 0.6622295352367195] -1.0 1.0
3 [0.95, 0.95, 0.4300953958935871, 0.6622295968901339] -1.0 1.0
5 [0.95, 0.9633111479844516, 0.6044862297285297, 0.6622296122264207] -1.0 1.0
4 [0.95, 0.9806222960455837, 0.6478659496449717, 0.8612296160413221] -1.0 1.0
4 [0.95, 0.9989284441257908, 0.6586566549741868, 0.9107308669902789] -1.0 1.0
```
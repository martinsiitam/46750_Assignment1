# 46750 - Optimization in Modern Power Systems: Assignment 1 Group 4

## Setup

From the folder containing `main.py`, run:

```bash
python -m pip install -r requirements.txt
```

A working **Gurobi license** is required (an academic license is suitable).

## 1. Run a base case

Run `python main.py --question CASE`, replacing `CASE` with:

| Case | Model |
| --- | --- |
| `Q1_caseA` | Linear utility; cheaper PV |
| `Q1_caseB` | Linear utility; more expensive PV |
| `Q2_linear` | Linear deviation discomfort, `c_L = 1.43` |
| `Q2_quadratic` | Quadratic discomfort, `c_Q = 1.0` |
| `Q2_linked_hours` | Quadratic + consecutive-shortfall discomfort |

For example:

```bash
python main.py --question Q2_quadratic
```

Each base-case run saves `inputs.png`, `schedule.png`, `duals.png`, and CSV/TXT results in `results/<CASE>/`

## 2. Change price profiles and get the same graphs

To use an alternative price profile (without modifying the original data):

```bash
python main.py --question Q2_linear --bus-params alternating_1 --show
```

Available profiles: `alternating_1`, `alternating_2`, `blocks_1`, `blocks_2`. Outputs go to a separate folder, such as `results/Q2_linear_alternating_1/`.

## 3. Reproduce the report's scenarios

Add `--scenarios` to run the actual report experiments, not the original starter-code examples:

| Command | Output |
| --- | --- |
| `python main.py --question Q1_caseA --scenarios` | Q1 Cases A and B (Table 6 in report) |
| `python main.py --question Q2_linear --scenarios` | Nine `c_L` values (Table 8 in report) |
| `python main.py --question Q2_quadratic --scenarios` | Six `c_Q` values (Table 9 in report) |
| `python main.py --question Q2_linked_hours --scenarios` | 3 models x 4 price profiles; paired averages (Table 14 in report) |

The runner prints comparison tables; scenario files, when saved by your version of `main.py`, are placed under `results/<CASE>/scenarios/`. 

**IMPORTANT**  Use **`--bus-params` alone** for a single price-profile run with its full graphs, do not combine it with `--scenarios`. Example `python main.py --question Q2_linked_hours --bus-params alternating_1`

**Reading the results:** The objective is *surplus*, not procurement cost. Q1 surplus = utility - procurement cost; Q2 surplus = -discomfort - procurement cost. At exact linear-cost ties, multiple schedules may have the same optimal surplus.

## Code layout

- `main.py` - command-line runs and report experiments
- `src/model.py` - optimization models and results
- `src/data_loader.py` - loads the JSON input data
- `src/scenarios.py` - creates modified inputs
- `src/plotting.py` - generates figures
- `data/` - original parameters and alternative price profiles
- `results/` - generated tables, solutions, and figures

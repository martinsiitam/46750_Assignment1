# 46750 - Optimization in Modern Power Systems: Assignment 1

## Requirements

- Packages listed in `requirements.txt` (including `gurobipy`, NumPy, pandas, and Matplotlib)
- A working **Gurobi license**

From the project folder containing `main.py`, install the packages:

```bash
python -m pip install -r requirements.txt
```

## Run the base cases

Run any of these commands from the project folder:

| Question | Command | Description |
| --- | --- | --- |
| Q1, Case A | `python main.py --question Q1_caseA` | PV cost below consumption utility |
| Q1, Case B | `python main.py --question Q1_caseB` | PV cost above consumption utility |
| Q2(b) | `python main.py --question Q2_linear` | Linear discomfort |
| Q2(c) | `python main.py --question Q2_quadratic` | Quadratic discomfort |
| Q2(e), bonus | `python main.py --question Q2_linked_hours` | Discomfort linked across hours |
| Q3 | `python main.py --question Q3` | Minimum daily energy requirement |
| Q3(g) | `python main.py --question Q3_battery` | Battery added |

Base-case runs print the optimal schedule and summary in the terminal. They also save result files and figures (such as `inputs.png`, `schedule.png`, and `duals.png`) under `results/<case>/`.

## Run the report scenarios

Add `--scenarios` to run the experiments associated with each question:

| Question | Command | What it runs |
| --- | --- | --- |
| Q2(b) | `python main.py --question Q2_linear --scenarios` | Sweep of linear discomfort coefficient `c_L` |
| Q2(c) | `python main.py --question Q2_quadratic --scenarios` | Sweep of quadratic discomfort coefficient `c_Q` |
| Q2(e) | `python main.py --question Q2_linked_hours --scenarios` | Linear, quadratic, and linked-hours models under alternating/block price profiles |
| Q3 | `python main.py --question Q3 --scenarios` | Q3 sensitivity analysis |
| Q3(g) |`python main.py --question Q3_battery --scenarios` | Q3(g) sensitivity analysis |


Scenario results are printed as comparison tables in the terminal; generated files are saved in the `results/` directory. **Surplus** is the maximized objective and is not the same as procurement cost.

### Run one Q2 price profile with full figures

Use `--bus-params` to select one of the Q2(e) price profiles:

```bash
python main.py --question Q2_linked_hours --bus-params alternating_1 --show
```

Available profiles: `alternating_1`, `alternating_2`, `blocks_1`, and `blocks_2`. This runs **one** profile and generates its base-case figures. Use `--scenarios` separately to run the complete comparison.

## Project files

- `main.py` — command-line interface and scenario runs
- `src/model.py` — optimization models and results
- `src/data_loader.py` — loads the input data
- `src/scenarios.py` — input modifications for sensitivity analyses
- `src/plotting.py` — graphs
- `data/` — provided model parameters and alternative price profiles
- `results/` — generated outputs

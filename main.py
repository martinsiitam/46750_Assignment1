"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results, LinearDiscomfortModel, QuadraticDiscomfortModel, minimumDailyLoadModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")
    if data.question == "Q2_linear":
        # 1.43 is the default
        # Values to try [0.5, 1.34, 1.42, 1.6, 2.0, 2.09, 2.11, 3.1]
        c_L = 1.43
        model = LinearDiscomfortModel(data, c_L=c_L).build()
    elif data.question == "Q2_quadratic":
        model = QuadraticDiscomfortModel(data, c_Q=1.0).build()
    elif data.question == "Q3":
        model = minimumDailyLoadModel(data, c_Q=1.0).build()
    else:
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
        # 1. Primal solution: one row per hour
        primal = results.hourly[["load", "pv", "import", "export"]]
        print("\nPrimal solution:")
        #print(primal.to_string())
        print(primal.to_string(float_format=lambda x: f"{0.0 if abs(x) < 1e-6 else x:.4f}"))

        # 2. Objective, procurement cost and utility
        financial = results.meta["financial_breakdown"]

        print(f"\nSurplus: {results.objective:.4f} DKK")
        print(f"Procurement cost: {financial['net_procurement_cost_DKK']:.4f} DKK")
        print(f"Utility: {financial['utility_DKK']:.4f} DKK")

        # 3. Dual variables required in Question 1(c)
        duals = results.hourly[[
            "dual_balance",
            "dual_pv_upper",
            "dual_load_lower",
            "dual_load_upper",
        ]]
        print("\nDual variables:")
        print(duals.to_string())
        # Hours at bounds
        load = results.hourly["load"]
        tol = 1e-6
        hours_at_bounds = ((abs(load - data.load_min_kWh) <= tol) | (abs(load - data.load_max_kWh) <= tol)).sum()

        print(f"Hours at load bounds: {hours_at_bounds}")
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

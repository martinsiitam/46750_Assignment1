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
from src.model import FlexibleConsumerModel, Results, LinearDiscomfortModel, QuadraticDiscomfortModel, minimumDailyLoadModel, LinkedHoursDiscomfortModel, batteryModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_load_preferences, shift_reference_load
from src.scenarios import set_fixed_price, set_alternate_prices, set_battery_preferences

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# CONFIGURATIONS
LINEAR_SWEEP = (0.50, 1.34, 1.42, 1.60, 2.00, 2.09, 2.11, 3.10) # 1.43 is default
QUADRATIC_SWEEP = (0.10, 0.50, 2.00, 5.00, 10.00) #  1.00 is default
PRICE_PROFILES = ("alternating_1", "alternating_2", "blocks_1", "blocks_2")

def build_model_for_case(data):
    """Build the correct optimization model for the loaded question data."""
    if data.question in ("Q1_caseA", "Q1_caseB"):
        return FlexibleConsumerModel(data).build()

    elif data.question == "Q2_linear":
        # 1.43 is the default
        # Values to try for sweep [0.5, 1.34, 1.42, 1.6, 2.0, 2.09, 2.11, 3.1]
        return LinearDiscomfortModel(data, c_L=1.43).build()

    elif data.question == "Q2_quadratic":
        # 1.0 is the default
        # Values to try for sweep [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        return QuadraticDiscomfortModel(data, c_Q=1.0).build()

    elif data.question == "Q2_linked_hours":
        return LinkedHoursDiscomfortModel(data, c_Q=1.0, c_P=1.0).build()
    
    elif data.question == "Q3":
        return minimumDailyLoadModel(data, c_Q=1.0).build()

    elif data.question == "Q3_battery":
        return batteryModel(data, c_Q=1.0).build()

    raise NotImplementedError(f"No model implemented for {data.question}")

def build_scenario_models(question: str, bus_params: str | None = None):
    """Build the correct optimization model for the loaded question data."""
    data = load_question(question, bus_params=bus_params)
    if question in ("Q1_caseA", "Q1_caseB"):
        return {"base": FlexibleConsumerModel(data).build()}
    elif question == "Q2_linear":
        models = {}
        for c_L in LINEAR_SWEEP:
            models[c_L] = LinearDiscomfortModel(data, c_L=c_L).build()
        return models
    elif question == "Q2_quadratic":
        models = {}
        for c_Q in QUADRATIC_SWEEP:
            models[c_Q] = QuadraticDiscomfortModel(data, c_Q=c_Q).build()
        return models
    elif question == "Q2_linked_hours": # Instead of sweep this has the different price profiles across the different models (model default kept as input)
        models = {}
        for profile in PRICE_PROFILES:
            # Linked hours
            data = load_question(question, bus_params=profile)
            models["linked hours " + profile] = LinkedHoursDiscomfortModel(data, c_Q=1.0, c_P=1.0).build()
            # Quadratic
            data = load_question("Q2_quadratic", bus_params=profile)
            models["quadratic " + profile] = QuadraticDiscomfortModel(data, c_Q=1.0).build()
            # Linear
            data = load_question("Q2_linear", bus_params=profile)
            models["linear " + profile] = LinearDiscomfortModel(data, c_L=1.43).build()
        return models
    elif question == "Q3":
        base = load_question(question, bus_params=bus_params)
        scenarios = {
                "base" : base,
                "lower daily minimum load - 22" : set_load_preferences(base, min_daily_energy_kWh = 22),
                "lower daily minimum load - 23" : set_load_preferences(base, min_daily_energy_kWh = 23),
                "lower daily minimum load - 30" : set_load_preferences(base, min_daily_energy_kWh = 30),
                "lower daily minimum load - 32" : set_load_preferences(base, min_daily_energy_kWh = 32),
                "lower daily minimum load - 32.3" : set_load_preferences(base, min_daily_energy_kWh = 32.3),
                "lower daily minimum load - 35" : set_load_preferences(base, min_daily_energy_kWh = 35),
                "higher minimum reference load 39.3" : shift_reference_load(base, 7/24),
                "higher minimum reference load 40" : shift_reference_load(base, 7.7/24),
                "higher minimum reference load 40.3" : shift_reference_load(base, 8/24),
                "higher minimum reference load 45" : shift_reference_load(base, 12.7/24),
                "higher minimum reference load 50" : shift_reference_load(base, 17.7/24),
                "higher minimum reference load 55" : shift_reference_load(base, 22.7/24),
                "higher minimum reference load 60" : shift_reference_load(base, 27.7/24),
                "higher minimum reference load 65" : shift_reference_load(base, 32.7/24)
        }
        models = {}
        for name, input_data in scenarios.items():
            models[name] = minimumDailyLoadModel(input_data, c_Q=1.0).build()
        return models
    elif question == "Q3_battery":
        base = load_question(question, bus_params=bus_params)
        scenarios = {
                "base" : base,
                "fixed prices" : set_fixed_price(base, 1.75),
                "alternating prices 1" : set_alternate_prices(base, 4),
                "alternating prices 2" : set_alternate_prices(base, 4, startHigh=False),
                "battery_capacity = 2" : set_battery_preferences(base, battery_capacity_kWh=2.0),
                "battery_capacity = 6" : set_battery_preferences(base, battery_capacity_kWh=6.0),

        }
        models = {}
        for name, input_data in scenarios.items():
            models[name] = batteryModel(input_data, c_Q=1.0).build()
        return models
    raise NotImplementedError(f"No model implemented for {question}")


def run_base_case(question: str, out: Path, show: bool, bus_params: str | None = None) -> Results | None:
    data = load_question(question, bus_params=bus_params)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")
    model = build_model_for_case(data)
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
        duals = results.hourly[["dual_balance", "dual_pv_upper", "dual_load_lower", "dual_load_upper"]]
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
    if bus_params is not None:
        out = out.with_name(out.name + "_" +  bus_params)
        results.question = results.question + "_" + bus_params
        out.mkdir(parents=True, exist_ok=True)
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path, bus_params: str | None = None) -> dict[str, Results]:
    scenario_models = build_scenario_models(question, bus_params=bus_params)
    runs: dict[str, Results] = {}
    is_sweep = question in ("Q2_linear", "Q2_quadratic")
    if is_sweep:
        coefficient_name = "c_L" if question == "Q2_linear" else "c_Q"
        print(
            f"\n{coefficient_name:>7} "
            f"{'Procurement':>13} "
            f"{'Discomfort':>12} "
            f"{'Consumption':>13} "
            f"{'Deviation':>11} "
            f"{'Bounds':>7} "
            f"{'Surplus':>12}")
        print("-" * 87)
    if question == "Q2_linked_hours":
        print(
            f"\n{'Scenario':<30} "
            f"{'Surplus [DKK]':>14} "
            f"{'Import [kWh]':>14} "
            f"{'Export [kWh]':>14}"
        )
        print("-" * 78)
        
    if question == "Q3":
        print("\nQuestion 3 - Scenario Analysis")
        print(
            f"{'Scenario':<35}"
            f"{'Objective [DKK]':>18}"
            f"{'Minimum daily load dual':>11}"
            f"{'utility [DKK]':>17}"
        )
        print("-" * 87)
    if question == "Q3_battery":
        print("\nQuestion 3g - Scenario Analysis")
        print(
            f"{'Scenario':<35}"
            f"{'Objective [DKK]':>18}"
            f"{'utility [DKK]':>17}"
        )
    for name, model in sorted(scenario_models.items(), key=lambda x: str(x[0]).lower()): # sorted for easier viewing later
        results = model.solve()
        results.save(out, tag=name)
        runs[name] = results
        if is_sweep:
            # Financial results
            financial = results.meta["financial_breakdown"]
            procurement = financial["net_procurement_cost_DKK"]
            discomfort = financial["disutility_DKK"]
            surplus = results.objective
            # Daily consumption
            load = results.hourly["load"]
            consumption = load.sum()
            # Total absolute deviation from the reference
            reference = results.hourly["reference_load"]
            deviation = (load - reference).abs().sum()
            # Hours where a load bound is binding
            data = model.data
            tolerance = 1e-6
            at_lower = (load - data.load_min_kWh).abs() < tolerance
            at_upper = (load - data.load_max_kWh).abs() < tolerance
            binding_hours = int((at_lower | at_upper).sum())

            # Coefficient used in this scenario
            coefficient = model.c_L if question == "Q2_linear" else model.c_Q
            print(
                f"{coefficient:7.2f} "
                f"{procurement:13.4f} "
                f"{discomfort:12.4f} "
                f"{consumption:13.2f} "
                f"{deviation:11.2f} "
                f"{binding_hours:7d} "
                f"{surplus:12.4f}"
            )
        if question == "Q2_linked_hours":
            print(
                f"{name:<30} "
                f"{results.objective:>14.2f} "
                f"{results.hourly['import'].sum():>14.1f} "
                f"{results.hourly['export'].sum():>14.1f}"
            )
        if question == "Q3":
            print(
                f"{name:<35}"
                f"{results.objective:>18.2f}"
                f"{results.duals["minimum_daily_load"]:>17.2f}"
                f"{results.meta["financial_breakdown"]["utility_DKK"]:>18.2f}"
            )
        if question == "Q3_battery":
            print(
                f"{name:<35}"
                f"{results.objective:>18.2f}"
                f"{results.meta["financial_breakdown"]["utility_DKK"]:>18.2f}"
            )
    if question == "Q2_linked_hours":
        print("\nAverage results by price pattern")
        print(
            f"{'Model':<16} "
            f"{'Price pattern':<15} "
            f"{'Surplus [DKK]':>15} "
            f"{'Deviation [kWh]':>18}"
        )
        print("-" * 67)
        for model_name in ("linear", "linked hours", "quadratic"):
            for pattern, prefix in (
                ("Alternating", "alternating"),
                ("Blocks", "blocks")):

                pair = [runs[f"{model_name} {prefix}_{i}"] for i in (1, 2)]
                avg_surplus = sum(r.objective for r in pair) / 2
                avg_deviation = sum((r.hourly["load"] - r.hourly["reference_load"]).abs().sum() for r in pair) / 2
                print(
                    f"{model_name:<16} "
                    f"{pattern:<15} "
                    f"{avg_surplus:>15.4f} "
                    f"{avg_deviation:>18.4f}"
                )    
    
    plot_scenario_comparison(runs,"objective", save_to=out / "scenarios_surplus.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--bus-params",type=str, default=None, help="Alternative bus parameter JSON file") # allows to set "alternating" or "blocks". Used for Q2
    
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")
    bus_param = args.bus_params
    if bus_param is not None:
        if bus_param not in ["alternating_1", "alternating_2","blocks_1", "blocks_2"] :
            raise ValueError("To use custom --bus-params for 2.E.iv then it must be either 'alternating_1', 'alternating_2' or 'blocks_1', 'blocks_2'")
    base = run_base_case(args.question, out, args.show, bus_params = bus_param)
    if args.scenarios and base is not None:
        if bus_param is not None:
            out = out.with_name(out.name + "_" +  bus_param)
            out.mkdir(parents=True, exist_ok=True)
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

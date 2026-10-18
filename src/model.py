"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Declare decision variables, objective and constraints.

        TODO (Question 1): complete this method with the variables, objective and constraints
        of the problem you formulated in Question 1. Keep the naming pattern below so
        that ``solve()`` can return the primal and dual values automatically.
        """
        d, m, T = self.data, self.m, self.T
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        # --- Decision variables --------------------------------------------------------
        # TODO: identify and declare the decision variables of your formulation.
        # Store every variable family in self.var["<name>"]: solve() then returns its hourly
        # values automatically as a column of results.hourly.
        # Pattern for hourly variables (one per hour):
        #   self.var["<name>"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Pattern for a single (daily) variable:
        #   self.var["<name>"] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="<name>")
        # Notes:
        # * gurobipy indexes the names automatically: name="<name>" in addVars(T, ...) creates
        #   <name>[0], <name>[1], ..., <name>[23] - no need to build per-hour names yourself.
        # * vtype: the same format takes GRB.BINARY or GRB.INTEGER if you ever need them (the
        #   problem then becomes a MILP and dual values are no longer defined; solve() skips them).
        # * lb defaults to 0 in gurobipy: a free variable needs an explicit lb=-GRB.INFINITY, and
        #   a bound you want a dual for must be an explicit constraint, not lb=/ub= (see the README).
        # * naming the families "import", "export", "load", "pv" makes the standard plots of
        #   src/plotting.py work out of the box.

        for name in ("load", "pv", "import", "export"):
            self.var[name] = m.addVars(
                T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name
            )
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]

        # --- Objective ---------------------------------------------------------------
        # TODO: express the objective function and its direction (GRB.MINIMIZE or GRB.MAXIMIZE):
        #   m.setObjective(gp.quicksum(<expression in t> for t in T), <direction>)
        # The input-data attributes (with units) are documented in src/data_loader.py (InputData).

        # Consumption utility - PV cost - import expenditure + export revenue.
        m.setObjective(
            gp.quicksum(
                    d.consumption_utility * load[t]
                    - d.pv_marginal_cost * pv[t]
                    - price_import[t] * imports[t]
                    + price_export[t] * exports[t]
                for t in T),
                GRB.MAXIMIZE,
        )



        # --- Constraints -------------------------------------------------------------
        # TODO: add the constraints of your formulation.
        # Pattern for hourly constraints (one per hour, duals returned as a 24-vector; names are
        # indexed automatically, like for the variables):
        #   self.con["<name>"] = m.addConstrs(
        #       (<lhs expression> - <rhs expression> <= 0 for t in T), name="<name>")
        # Pattern for a single constraint (dual returned as a scalar):
        #   self.con["<name>"] = m.addConstr(<lhs expression> - <rhs expression> <= 0, name="<name>")
        self.con["balance"] = m.addConstrs(
            (load[t] + exports[t] - imports[t] - pv[t] == 0 for t in T),
            name="balance",
        )

        self.con["load_lower"] = m.addConstrs(
            (-load[t] <= -d.load_min_kWh for t in T), name="load_lower"
        )

        self.con["load_upper"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T), name="load_upper"
        )

        self.con["pv_lower"] = m.addConstrs(
            (-pv[t] <= 0 for t in T), name="pv_lower"
        )

        self.con["pv_upper"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T), name="pv_upper"
        )

        self.con["export_lower"] = m.addConstrs(
            (-exports[t] <= 0 for t in T), name="export_lower"
        )

        self.con["import_lower"] = m.addConstrs(
            (-imports[t] <= 0 for t in T), name="import_lower"
        )

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        meta = {"scalar_variables": scalars}
        # Q1 financial breakdown
        if d.consumption_utility is not None:
            price_import = d.energy_price + d.import_tariff
            price_export = d.energy_price - d.export_tariff

            hourly["utility_DKK"] = d.consumption_utility * hourly["load"]
            hourly["pv_cost_DKK"] = d.pv_marginal_cost * hourly["pv"]
            hourly["import_cost_DKK"] = price_import * hourly["import"]
            hourly["export_revenue_DKK"] = price_export * hourly["export"]
            hourly["net_procurement_cost_DKK"] = (
                hourly["pv_cost_DKK"] + hourly["import_cost_DKK"]
                - hourly["export_revenue_DKK"]
            )
            hourly["surplus_DKK"] = (
                hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"]
            )
            financial_columns = [
                "utility_DKK", "pv_cost_DKK", "import_cost_DKK",
                "export_revenue_DKK", "net_procurement_cost_DKK", "surplus_DKK",
            ]
            meta["financial_breakdown"] = {
                name: float(hourly[name].sum()) for name in financial_columns
            }


        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta=meta,
        )

class LinearDiscomfortModel(FlexibleConsumerModel):
    """Question 2.b model"""
    def __init__(self, data: InputData, *, c_L: float, name: str = "linear_discomfort", verbose: bool = False):
        """Initialize with case data and c_L in DKK/kWh."""
        c_L = float(c_L)
        if not np.isfinite(c_L) or c_L <= 0:
            raise ValueError("c_L must be finite and strictly positive.")
        if data.reference_load is None:
            raise ValueError("Q2(b) requires a reference load profile.")
        reference = np.asarray(data.reference_load, dtype=float)
        if reference.shape != (data.n_hours,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_load must contain one finite value per hour.")
        super().__init__(data, name=name, verbose=verbose)
        self.c_L = c_L
        
    def build(self) -> "LinearDiscomfortModel":
        d, m, T = self.data, self.m, self.T
        
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        # Decision variables
        for name in ("load", "pv", "import", "export"):
            self.var[name] = m.addVars(T, lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name=name)
        self.var["abs_deviation"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_deviation")
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]
        deviation = self.var["abs_deviation"]
        # Objective
        m.setObjective(
            gp.quicksum(
                -self.c_L * deviation[t]
                - d.pv_marginal_cost * self.var["pv"][t]
                - price_import[t] * self.var["import"][t]
                + price_export[t] * self.var["export"][t]
                for t in T
            ),
            GRB.MAXIMIZE,
        )
        
        # Physical constraints
        self.con["balance"] = m.addConstrs(
            (load[t] + exports[t] - imports[t] - pv[t] == 0
            for t in T),
            name="balance")
        self.con["load_lower"] = m.addConstrs(
            (-load[t] <= -d.load_min_kWh for t in T),
            name="load_lower")
        self.con["load_upper"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T),
            name="load_upper")
        self.con["pv_lower"] = m.addConstrs(
            (-pv[t] <= 0 for t in T),
            name="pv_lower")
        self.con["pv_upper"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T),
            name="pv_upper")
        self.con["import_lower"] = m.addConstrs(
            (-imports[t] <= 0 for t in T),
            name="import_lower")
        self.con["export_lower"] = m.addConstrs(
            (-exports[t] <= 0 for t in T),
            name="export_lower")
        
        # Absolute-deviation constraints
        self.con["deviation_positive"] = m.addConstrs(
            (load[t] - d.reference_load[t] <= deviation[t] 
             for t in T),
            name="deviation_positive")
        self.con["deviation_negative"] = m.addConstrs(
            (d.reference_load[t] - load[t] <= deviation[t] 
             for t in T),
            name="deviation_negative")
        m.update()
        return self
    
    def _add_financial_breakdown(self, hourly: pd.DataFrame, meta: dict) -> None:
        """Add hourly and daily Q2 costs (DKK) and energy deviations (kWh)."""
        d = self.data
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        hourly["deviation_kWh"] = hourly["load"] - hourly["reference_load"]
        hourly["absolute_deviation_kWh"] = hourly["deviation_kWh"].abs()
        hourly["disutility_DKK"] = self.c_L * hourly["absolute_deviation_kWh"]
        hourly["utility_DKK"] = -hourly["disutility_DKK"]
        hourly["pv_cost_DKK"] = d.pv_marginal_cost * hourly["pv"]
        hourly["import_cost_DKK"] = price_import * hourly["import"]
        hourly["export_revenue_DKK"] = price_export * hourly["export"]
        hourly["net_procurement_cost_DKK"] = (hourly["pv_cost_DKK"] + hourly["import_cost_DKK"] - hourly["export_revenue_DKK"])
        hourly["surplus_DKK"] = (hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"])
        financial_columns = ["disutility_DKK", "utility_DKK", "pv_cost_DKK", "import_cost_DKK", "export_revenue_DKK", "net_procurement_cost_DKK", "surplus_DKK"]
        meta["financial_breakdown"] = {name: float(hourly[name].sum()) for name in financial_columns}
        meta["c_L"] = self.c_L
    
    def _extract_results(self, status: str) -> Results:
        """Extract the solution and add linear discomfort costs."""
        results = super()._extract_results(status)
        self._add_financial_breakdown(results.hourly, results.meta)
        return results

class QuadraticDiscomfortModel(FlexibleConsumerModel):
    """Question 2.c model"""

    def __init__(self, data: InputData,*,c_Q: float, name: str = "quadratic_discomfort", verbose: bool = False):
        c_Q = float(c_Q)
        if not np.isfinite(c_Q) or c_Q <= 0:
            raise ValueError("c_Q must be finite and strictly positive.")
        if data.reference_load is None:
            raise ValueError("The quadratic discomfort model requires a reference load.")
        reference = np.asarray(data.reference_load, dtype=float)
        if (reference.shape != (data.n_hours,) or not np.all(np.isfinite(reference))):
            raise ValueError("reference_load must contain one finite value per hour.")
        super().__init__(data, name=name, verbose=verbose)
        self.c_Q = c_Q

    def build(self) -> "QuadraticDiscomfortModel":
        """Build the surplus-maximizing QP for one-hour periods."""
        d, m, T = self.data, self.m, self.T
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff

        # Decision variables
        for name in ("load", "pv", "import", "export"):
            self.var[name] = m.addVars(T,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name=name)
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]

        # Objective: negative discomfort and net procurement cost
        m.setObjective(
            gp.quicksum(
                -self.c_Q * (load[t] - d.reference_load[t]) ** 2
                - d.pv_marginal_cost * pv[t]
                - price_import[t] * imports[t]
                + price_export[t] * exports[t]
                for t in T),
            GRB.MAXIMIZE)

        # Physical constraints
        self.con["balance"] = m.addConstrs((load[t] + exports[t] - imports[t] - pv[t] == 0 for t in T),name="balance")
        self.con["load_lower"] = m.addConstrs((-load[t] <= -d.load_min_kWh for t in T),name="load_lower")
        self.con["load_upper"] = m.addConstrs((load[t] <= d.load_max_kWh for t in T),name="load_upper")
        self.con["pv_lower"] = m.addConstrs((-pv[t] <= 0 for t in T),name="pv_lower")
        self.con["pv_upper"] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T),name="pv_upper")
        self.con["import_lower"] = m.addConstrs((-imports[t] <= 0 for t in T),name="import_lower")
        self.con["export_lower"] = m.addConstrs((-exports[t] <= 0 for t in T), name="export_lower")
        m.update()
        return self

    def _add_financial_breakdown(self,hourly: pd.DataFrame,meta: dict) -> None:
        """Add hourly and daily costs in DKK and deviations in kWh."""
        d = self.data
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        hourly["deviation_kWh"] = (hourly["load"] - hourly["reference_load"])
        hourly["absolute_deviation_kWh"] = (hourly["deviation_kWh"].abs())
        hourly["disutility_DKK"] = (self.c_Q * hourly["deviation_kWh"] ** 2)
        hourly["utility_DKK"] = -hourly["disutility_DKK"]
        hourly["pv_cost_DKK"] = (d.pv_marginal_cost * hourly["pv"])
        hourly["import_cost_DKK"] = (price_import * hourly["import"])
        hourly["export_revenue_DKK"] = (price_export * hourly["export"])
        hourly["net_procurement_cost_DKK"] = (hourly["pv_cost_DKK"] + hourly["import_cost_DKK"] - hourly["export_revenue_DKK"])
        hourly["surplus_DKK"] = (hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"])
        financial_columns = [
            "utility_DKK",
            "disutility_DKK",
            "pv_cost_DKK",
            "import_cost_DKK",
            "export_revenue_DKK",
            "net_procurement_cost_DKK",
            "surplus_DKK"
        ]
        meta["financial_breakdown"] = {name: float(hourly[name].sum()) for name in financial_columns}
        meta["c_Q"] = self.c_Q
        
    def _extract_results(self, status: str) -> Results:
        """Extract the solution and add quadratic discomfort costs."""
        results = super()._extract_results(status)
        self._add_financial_breakdown(results.hourly, results.meta)
        return results


class minimumDailyLoadModel(FlexibleConsumerModel):
    """Question 3 model"""

    def __init__(self, data: InputData,*,c_Q: float, name: str = "minimum_daily_load_model", verbose: bool = False):
        c_Q = float(c_Q)
        if not np.isfinite(c_Q) or c_Q <= 0:
            raise ValueError("c_Q must be finite and strictly positive.")
        if data.reference_load is None:
            raise ValueError("The minimum daily load model requires a reference load.")
        if data.min_daily_energy_kWh is None:
            raise ValueError("The minimum daily load model requires a minimum daily load.")
        reference = np.asarray(data.reference_load, dtype=float)
        if (reference.shape != (data.n_hours,) or not np.all(np.isfinite(reference))):
            raise ValueError("reference_load must contain one finite value per hour.")
        super().__init__(data, name=name, verbose=verbose)
        self.c_Q = c_Q

    def build(self) -> "minimumDailyLoadModel":
        d, m, T = self.data, self.m, self.T
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff

        # Decision variables
        for name in ("load", "pv", "import", "export"):
            self.var[name] = m.addVars(T,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name=name)
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]

        # Objective: negative discomfort and net procurement cost
        m.setObjective(
            gp.quicksum(
                -self.c_Q * (load[t] - d.reference_load[t]) ** 2
                - d.pv_marginal_cost * pv[t]
                - price_import[t] * imports[t]
                + price_export[t] * exports[t]
                for t in T),
            GRB.MAXIMIZE)

        # Physical constraints
        self.con["balance"] = m.addConstrs((load[t] + exports[t] - imports[t] - pv[t] == 0 for t in T),name="balance")
        self.con["load_lower"] = m.addConstrs((-load[t] <= -d.load_min_kWh for t in T),name="load_lower")
        self.con["load_upper"] = m.addConstrs((load[t] <= d.load_max_kWh for t in T),name="load_upper")
        self.con["pv_lower"] = m.addConstrs((-pv[t] <= 0 for t in T),name="pv_lower")
        self.con["pv_upper"] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T),name="pv_upper")
        self.con["import_lower"] = m.addConstrs((-imports[t] <= 0 for t in T),name="import_lower")
        self.con["export_lower"] = m.addConstrs((-exports[t] <= 0 for t in T), name="export_lower")
        self.con["minimum_daily_load"] = m.addConstr((gp.quicksum(load[t] for t in T) >= d.min_daily_energy_kWh), name = "minimum_daily_load")
        m.update()
        return self

    def _add_financial_breakdown(self,hourly: pd.DataFrame,meta: dict) -> None:
        """Add hourly and daily costs in DKK and deviations in kWh."""
        d = self.data
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        hourly["deviation_kWh"] = (hourly["load"] - hourly["reference_load"])
        hourly["absolute_deviation_kWh"] = (hourly["deviation_kWh"].abs())
        hourly["disutility_DKK"] = (self.c_Q * hourly["deviation_kWh"] ** 2)
        hourly["utility_DKK"] = -hourly["disutility_DKK"]
        hourly["pv_cost_DKK"] = (d.pv_marginal_cost * hourly["pv"])
        hourly["import_cost_DKK"] = (price_import * hourly["import"])
        hourly["export_revenue_DKK"] = (price_export * hourly["export"])
        hourly["net_procurement_cost_DKK"] = (hourly["pv_cost_DKK"] + hourly["import_cost_DKK"] - hourly["export_revenue_DKK"])
        hourly["surplus_DKK"] = (hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"])
        financial_columns = [
            "utility_DKK",
            "disutility_DKK",
            "pv_cost_DKK",
            "import_cost_DKK",
            "export_revenue_DKK",
            "net_procurement_cost_DKK",
            "surplus_DKK"
        ]
        meta["financial_breakdown"] = {name: float(hourly[name].sum()) for name in financial_columns}
        meta["c_Q"] = self.c_Q
        meta["minimum_daily_load"] = self.data.min_daily_energy_kWh
        meta["sum_of_reference_loads"] = np.sum(self.data.reference_load)
        
    def _extract_results(self, status: str) -> Results:
        """Extract the solution and add quadratic discomfort costs."""
        results = super()._extract_results(status)
        self._add_financial_breakdown(results.hourly, results.meta)
        return results
        
class batteryModel(FlexibleConsumerModel):
    """Question 3.G model"""

    def __init__(self, data: InputData,*,c_Q: float, name: str = "battery_model", verbose: bool = False):
        c_Q = float(c_Q)
        if not np.isfinite(c_Q) or c_Q <= 0:
            raise ValueError("c_Q must be finite and strictly positive.")
        if data.reference_load is None:
            raise ValueError("The battery model requires a reference load.")
        if data.min_daily_energy_kWh is None:
            raise ValueError("The battery model requires a minimum daily load.")
        reference = np.asarray(data.reference_load, dtype=float)
        if (reference.shape != (data.n_hours,) or not np.all(np.isfinite(reference))):
            raise ValueError("reference_load must contain one finite value per hour.")
        if data.battery_capacity_kWh is None:
            raise ValueError("The battery model requires a battery capacity")
        if data.battery_max_charge_kW is None:
            raise ValueError("The battery model requires a battery max charge power")
        if data.battery_max_discharge_kW is None:
            raise ValueError("The battery model requires a battery max discharge power")
        if data.battery_charging_efficiency is None:
            raise ValueError("The battery model requires a battery charging efficiency")
        if data.battery_discharging_efficiency is None:
            raise ValueError("The battery model requires a battery discharging efficiency")
        if data.battery_initial_soc_kWh is None:
            raise ValueError("The battery model requires a battery initial state of charge")
        super().__init__(data, name=name, verbose=verbose)
        self.c_Q = c_Q

    def build(self) -> "batteryModel":
        d, m, T = self.data, self.m, self.T
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff

        # Decision variables
        for name in ("load", "pv", "import", "export", "battery_soc", "battery_chg", "battery_dis"):
            self.var[name] = m.addVars(T,lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS, name=name)
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]
        battery_soc = self.var["battery_soc"]
        battery_chg = self.var["battery_chg"]
        battery_dis = self.var["battery_dis"]

        # Objective: negative discomfort and net procurement cost
        m.setObjective(
            gp.quicksum(
                -self.c_Q * (load[t] - d.reference_load[t]) ** 2
                - d.pv_marginal_cost * pv[t]
                - price_import[t] * imports[t]
                + price_export[t] * exports[t]
                for t in T),
            GRB.MAXIMIZE)

        # Physical constraints
        self.con["balance"] = m.addConstrs((load[t] + exports[t] + battery_chg[t]- imports[t] - pv[t] - battery_dis[t]== 0 for t in T),name="balance")
        self.con["load_lower"] = m.addConstrs((-load[t] <= -d.load_min_kWh for t in T),name="load_lower")
        self.con["load_upper"] = m.addConstrs((load[t] <= d.load_max_kWh for t in T),name="load_upper")
        self.con["pv_lower"] = m.addConstrs((-pv[t] <= 0 for t in T),name="pv_lower")
        self.con["pv_upper"] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T),name="pv_upper")
        self.con["import_lower"] = m.addConstrs((-imports[t] <= 0 for t in T),name="import_lower")
        self.con["export_lower"] = m.addConstrs((-exports[t] <= 0 for t in T), name="export_lower")
        self.con["minimum_daily_load"] = m.addConstr((gp.quicksum(load[t] for t in T) >= d.min_daily_energy_kWh), name = "minimum_daily_load")
        m.update()
        self.con["battery_capacity_lower"] = m.addConstrs((battery_soc[t] >= 0 for t in T), name = "battery_capacity_lower")
        self.con["battery_capacity_upper"] = m.addConstrs((battery_soc[t] <= d.battery_capacity_kWh for t in T), name = "battery_capacity_upper")
        self.con["battery_charging_lower"] = m.addConstrs((battery_chg[t] >= 0 for t in T), name = "battery_charging_lower")
        self.con["battery_charging_upper"] = m.addConstrs((battery_chg[t] <= d.battery_max_charge_kW for t in T), name = "battery_charging_upper")
        self.con["battery_discharge_lower"] = m.addConstrs((battery_dis[t] >= 0 for t in T), name = "battery_discharge_lower")
        self.con["battery_discharge_upper"] = m.addConstrs((battery_dis[t] <= d.battery_max_discharge_kW for t in T), name = "battery_discharge_upper")
        self.con["battery_capacity_coupling"] = m.addConstrs((battery_soc[t] == battery_soc[t-1] + d.battery_charging_efficiency * battery_chg[t] - (1/d.battery_discharging_efficiency) * battery_dis[t] for t in range(1,24)), name = "battery_capacity_coupling")
        self.con["battery_capacity_coupling"][0] = m.addConstr((battery_soc[0] == d.battery_initial_soc_kWh + d.battery_charging_efficiency * battery_chg[0] - (1/d.battery_discharging_efficiency) * battery_dis[0]), name = "battery_capacity_coupling")
        return self

    def _add_financial_breakdown(self,hourly: pd.DataFrame,meta: dict) -> None:
        """Add hourly and daily costs in DKK and deviations in kWh."""
        d = self.data
        price_import = d.energy_price + d.import_tariff
        price_export = d.energy_price - d.export_tariff
        hourly["deviation_kWh"] = (hourly["load"] - hourly["reference_load"])
        hourly["absolute_deviation_kWh"] = (hourly["deviation_kWh"].abs())
        hourly["disutility_DKK"] = (self.c_Q * hourly["deviation_kWh"] ** 2)
        hourly["utility_DKK"] = -hourly["disutility_DKK"]
        hourly["pv_cost_DKK"] = (d.pv_marginal_cost * hourly["pv"])
        hourly["import_cost_DKK"] = (price_import * hourly["import"])
        hourly["export_revenue_DKK"] = (price_export * hourly["export"])
        hourly["net_procurement_cost_DKK"] = (hourly["pv_cost_DKK"] + hourly["import_cost_DKK"] - hourly["export_revenue_DKK"])
        hourly["surplus_DKK"] = (hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"])
        financial_columns = [
            "utility_DKK",
            "disutility_DKK",
            "pv_cost_DKK",
            "import_cost_DKK",
            "export_revenue_DKK",
            "net_procurement_cost_DKK",
            "surplus_DKK"
        ]
        meta["financial_breakdown"] = {name: float(hourly[name].sum()) for name in financial_columns}
        meta["c_Q"] = self.c_Q
        meta["minimum_daily_load"] = self.data.min_daily_energy_kWh
        meta["sum_of_reference_loads"] = np.sum(self.data.reference_load)
        
    def _extract_results(self, status: str) -> Results:
        """Extract the solution and add quadratic discomfort costs."""
        results = super()._extract_results(status)
        self._add_financial_breakdown(results.hourly, results.meta)
        return results

     
class LinkedHoursDiscomfortModel(QuadraticDiscomfortModel):
    """Quadratic discomfort plus consecutive-shortfall penalty."""     
    def __init__(self, data: InputData, *, c_Q: float = 1.0, c_P: float = 1.0,name: str = "nonseparable_discomfort", verbose: bool = False):
        super().__init__(data, c_Q=c_Q, name=name, verbose=verbose)
        c_P = float(c_P)
        if not np.isfinite(c_P) or c_P <= 0:
            raise ValueError("c_P must be finite and positive.")
        self.c_P = c_P
   
    def build(self) -> "LinkedHoursDiscomfortModel":
        # Reuse Q2(c)'s quadratic objective and physical constraints.
        super().build()
        d, m = self.data, self.m
        n = d.n_hours
        load = self.var["load"]
        s = self.var["shortfall"] = m.addVars(self.T, lb=0.0, vtype=GRB.CONTINUOUS, name="shortfall")
        self.con["shortfall_lower"] = m.addConstrs((s[t] >= d.reference_load[t] - load[t] for t in self.T),name="shortfall_lower")
        # Boundary conditions s[-1] = s[n] = 0.
        persistence_cost = 0.5 * self.c_P * (s[0] * s[0] + gp.quicksum((s[t - 1] + s[t]) * (s[t - 1] + s[t]) for t in range(1, n)) + s[n - 1] * s[n - 1])
        m.setObjective(m.getObjective() - persistence_cost, GRB.MAXIMIZE)
        m.update()
        return self
    
    def _add_financial_breakdown(self, hourly: pd.DataFrame, meta: dict) -> None:
        # Q2(c) already computes the same PV/import/export costs and deviations.
        super()._add_financial_breakdown(hourly, meta)
        s = hourly["shortfall"].to_numpy(dtype=float)
        prev = np.concatenate(([0.0], s[:-1]))
        following = np.concatenate((s[1:], [0.0]))

        # Expand the reported penalty:
        # c_P/2 * sum((s[t-1]+s[t])^2, t=0..n)
        #   = c_P * sum(s[t]^2) + c_P * sum(s[t-1]*s[t], t=1..n-1).
        # Each neighbouring product is shared equally by its two hours for
        # reporting; the summed hourly entries equal the exact daily penalty.
        hourly["quadratic_disutility_DKK"] = self.c_Q * hourly["deviation_kWh"] ** 2
        hourly["persistence_disutility_DKK"] = self.c_P * (s**2 + 0.5 * s * (prev + following))
        hourly["disutility_DKK"] = (hourly["quadratic_disutility_DKK"] + hourly["persistence_disutility_DKK"])
        hourly["utility_DKK"] = -hourly["disutility_DKK"]
        hourly["surplus_DKK"] = (hourly["utility_DKK"] - hourly["net_procurement_cost_DKK"])
        financial = meta["financial_breakdown"]
        for key in ("disutility_DKK", "utility_DKK", "surplus_DKK"):
            financial[key] = float(hourly[key].sum())
        financial["quadratic_disutility_DKK"] = float(hourly["quadratic_disutility_DKK"].sum())
        financial["persistence_disutility_DKK"] = float(hourly["persistence_disutility_DKK"].sum())
        meta["c_P"] = self.c_P

    def _extract_results(self, status: str) -> Results:
        results = super()._extract_results(status)
        results.question = "Q2_linked_hours"
        return results
   
_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")

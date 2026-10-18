"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data_loader import InputData


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Multiply hourly prices by ``factor``. With ``keep_mean=True`` only the spread around the
    mean is scaled, which isolates the effect of price variability from the effect of price level."""
    p = data.energy_price
    new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
    return replace(data, energy_price=np.clip(new_p, 0.0, None))   # prices stay non-negative

def scale_reference_load(data: InputData, factor: float) -> InputData:
    """Multiplies the reference load by ``factor``."""
    p = data.reference_load
    new_p = p * factor
    return replace(data, reference_load=np.clip(new_p, 0.0, None))   # prices stay non-negative

def shift_reference_load(data: InputData, delta: float) -> InputData:
    """Shifts the reference load by ``delta``."""
    return replace(data, reference_load=np.clip(data.reference_load + delta, 0.0, None))

def shift_prices(data: InputData, delta: float) -> InputData:
    """Add ``delta`` DKK/kWh to every hourly price (changes the mean, keeps the spread)."""
    return replace(data, energy_price=np.clip(data.energy_price + delta, 0.0, None))

def set_fixed_price(data: InputData, price:float) -> InputData:
    """Sets all the hourly prices to a single price"""
    return replace(data, energy_price=np.array([price for t in data.hours]))

def set_alternate_prices(data: InputData, price:float, startHigh : bool = True) -> InputData:
    """Sets alternate high and low prices, with high price = float, and low price = 0.5"""
    prices = [0.5 for t in data.hours]
    if startHigh:
        for t in data.hours:
            if t % 2 == 0:
                prices[t] = price
    else:
        for t in data.hours:
            if t % 2 == 1:
                prices[t] = price
    return replace(data, energy_price=np.array(prices))
    


def roll_prices(data: InputData, hours: int) -> InputData:
    """Shift the price profile in time by ``hours`` (positive = later). Useful to study the
    temporal correlation between the price peak and the PV peak."""
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(data: InputData, import_tariff: float | None = None, export_tariff: float | None = None) -> InputData:
    """Override the import and/or export grid tariff (DKK/kWh)."""
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Scale the available PV production (e.g. 0.5 for a cloudy day, 0.0 for no PV)."""
    profile = np.clip(data.pv_profile * factor, 0.0, 1.0)
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Override the hourly load bounds and/or the minimum daily energy requirement."""
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=data.min_daily_energy_kWh if min_daily_energy_kWh is None else min_daily_energy_kWh,
    )

def set_battery_preferences(
    data: InputData,
    battery_capacity_kWh: float | None = None,
    battery_max_charge_kW: float | None = None,
    battery_max_discharge_kW: float | None = None,
    battery_charging_efficiency: float | None = None,
    battery_discharging_efficiency: float | None = None,
    battery_initial_soc_kWh: float | None = None,
    battery_final_soc_kWh: float | None = None,
) -> InputData:
    """Override the characteristics of the battery."""
    return replace(
        data,
        battery_capacity_kWh = data.battery_capacity_kWh if battery_capacity_kWh is None else battery_capacity_kWh,
        battery_max_charge_kW = data.battery_max_charge_kW if battery_max_charge_kW is None else battery_max_charge_kW,
        battery_max_discharge_kW = data.battery_max_discharge_kW if battery_max_discharge_kW is None else battery_max_discharge_kW,
        battery_charging_efficiency = data.battery_charging_efficiency if battery_charging_efficiency is None else battery_charging_efficiency,
        battery_discharging_efficiency = data.battery_discharging_efficiency if battery_discharging_efficiency is None else battery_discharging_efficiency,
        battery_initial_soc_kWh = data.battery_initial_soc_kWh if battery_initial_soc_kWh is None else battery_initial_soc_kWh,
        battery_final_soc_kWh = data.battery_final_soc_kWh if battery_final_soc_kWh is None else battery_final_soc_kWh,
    )



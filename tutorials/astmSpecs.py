"""Compare FuelLib predictions against ASTM D7566 specification limits."""

import numpy as np

import fuellib as fl

fuel_name = "posf10325"
fuel_dir = None
use_ref_props = True
viscosity_correlation = "Nannoolal"  # "Nannoolal" or "Dutt"

fuel = fl.Fuel(fuel_name, fuelDataDir=fuel_dir, useRefProperties=use_ref_props)
Y = fuel.Y_0

# ASTM D7566 properties
astm_d7566_props = {
    "Density": {
        "min_value": fl.Units.Quantity([775], "kg/m^3"),
        "max_value": fl.Units.Quantity([840], "kg/m^3"),
        "at_temperature": fl.Units.Quantity([15], "celsius"),
    },
    "Flash Point": {
        "min_value": fl.Units.Quantity([38], "celsius"),
    },
    "Freezing Point": {
        "max_value": fl.Units.Quantity([-40], "celsius"),
        },
    "Viscosity": {
        "max_value": fl.Units.Quantity([8.0, 12.0], "mm^2/s"),
        "at_temperature": fl.Units.Quantity([-20, -40], "celsius"),
    },
    "YSI": {
        "value": fl.Units.Quantity([129.49], "dimensionless"),
    },
    "Heat of Combustion": {
        "min_value": fl.Units.Quantity([42.8], "MJ/kg"),
    },
    "DCN": {
            "value": fl.Units.Quantity([48.8], "dimensionless"),
            # Note the target DCN is the "best case" from Edwards 2020 Table 17
    },
}

# Map each ASTM property to a FuelLib prediction
predictors = {
    "Density": lambda T: fuel.mixture_density(Y, T),
    "Flash Point": lambda T: fuel.mixture_flash_point(Y),
    "Freezing Point": lambda T: fuel.mixture_freeze_point(Y),
    "Viscosity": lambda T: fuel.mixture_kinematic_viscosity(Y, T, correlation=viscosity_correlation),
    "YSI": lambda T: fuel.mixture_yield_sooting_index(Y),
    "Heat of Combustion": lambda T: fuel.mixture_heat_of_combustion(Y),
    "DCN": lambda T: fuel.mixture_derived_cetane_number(Y),
}


def _as_list(quantity, n):
    """Return `quantity` as a list of n scalar Quantities (None entries if missing)."""
    if quantity is None:
        return [None] * n
    values = list(np.atleast_1d(quantity.magnitude))
    if len(values) == 1:
        values = values * n
    if len(values) != n:
        raise ValueError(f"Expected 1 or {n} entries, got {len(values)}.")
    return [fl.Units.Quantity(float(v), quantity.units) for v in values]


def _fmt(quantity, units):
    """Format a Quantity in `units`, or '-' if None."""
    if quantity is None:
        return "-"
    return f"{quantity.to(units).magnitude:.2f}"


def evaluate(prop, criteria):
    """Yield one result row per temperature for a property."""
    if prop not in predictors:
        raise KeyError(f"No FuelLib predictor defined for {prop!r}.")

    min_q = criteria.get("min_value")
    max_q = criteria.get("max_value")
    val_q = criteria.get("value")
    temp_q = criteria.get("at_temperature")

    given = [q for q in (min_q, max_q, val_q) if q is not None]
    if not given:
        raise ValueError(f"No min_value, max_value, or value specified for {prop}.")
    units = given[0].units

    n = max(
        len(np.atleast_1d(q.magnitude))
        for q in (min_q, max_q, val_q, temp_q)
        if q is not None
    )
    mins, maxs, vals, temps = (_as_list(q, n) for q in (min_q, max_q, val_q, temp_q))

    for lo, hi, val, T in zip(mins, maxs, vals, temps):
        pred = predictors[prop](T.to("K") if T is not None else None)
        pred = fl.Units.Quantity(float(np.squeeze(pred.to(units).magnitude)), units)

        checks = []
        if lo is not None:
            checks.append(pred >= lo)
        if hi is not None:
            checks.append(pred <= hi)
        status = "-" if not checks else ("PASS" if all(checks) else "FAIL")
        yield pred, lo, hi, val, T, units, status


# Check ASTM D7566 properties
header = (
    f"{'Property':<18}{'FuelLib':>12}{'Min':>10}{'Max':>10}{'Value':>10}"
    f"{'T':>12}  {'Units':<10}Status"
)
print(f"\nASTM D7566 properties for fuel: {fuel.name}\n")
print(header)
print("-" * len(header))
for prop, criteria in astm_d7566_props.items():
    for pred, lo, hi, val, T, units, status in evaluate(prop, criteria):
        T_str = "-" if T is None else f"{T.to('celsius').magnitude:.1f} C"
        print(
            f"{prop:<18}{_fmt(pred, units):>12}{_fmt(lo, units):>10}"
            f"{_fmt(hi, units):>10}{_fmt(val, units):>10}{T_str:>12}  "
            f"{f'{units:~P}':<12}{status}"
        )


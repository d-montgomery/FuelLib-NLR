"""Compare FuelLib predictions against ASTM D7566 specification limits."""

import numpy as np

import fuellib as fl
from fuellib.simulate.distillation import greenfield

fuel_name = "posf10325"
fuel_dir = None

use_ref_props = True
viscosity_corr = "Nannoolal"  # "Dutt" (Default) or "Nannoolal"
flash_point_corr = "Alibakhshi" # "Alibakhshi" (Default) or "Alqaheem"

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
    # Smoke point: meet option (1) OR option (2)
    #   (1) Smoke point >= 25 mm
    #   (2) Smoke point >= 18 mm AND naphthalenes <= 3 vol%
    "Smoke Point (1)": {
        "min_value": fl.Units.Quantity([25], "mm"),
    },
    "Smoke Point (2)": {
        "min_value": fl.Units.Quantity([18], "mm"),
    },
    "Naphthalenes (2)": {
        "max_value": fl.Units.Quantity([3.0], "percent"),
    },
    # Distillation temperature (D86). Entries with only "units" are report-only.
    "Dist. T10": {
        "max_value": fl.Units.Quantity([205], "celsius"),
    },
    "Dist. T50": {"units": "celsius"},
    "Dist. T90": {"units": "celsius"},
    "Dist. FBP": {
        "max_value": fl.Units.Quantity([300], "celsius"),
    },
    "Dist. T50 - T10": {
        "min_value": fl.Units.Quantity([15], "delta_degC"),
    },
    "Dist. T90 - T10": {
        "min_value": fl.Units.Quantity([40], "delta_degC"),
    },
}

# Groups of requirements: all entries in an option must pass (AND), and at least
# one option in a group must pass (OR).
astm_d7566_either_or = {
    "Smoke Point": [
        ["Smoke Point (1)"],
        ["Smoke Point (2)", "Naphthalenes (2)"],
    ],
}

_d86_cache = {}


def d86():
    """Simulate the D86 distillation curve once and reuse it."""
    if "result" not in _d86_cache:
        _d86_cache["result"] = greenfield.distill(fuel, Y)
    return _d86_cache["result"]


# Map each ASTM property to a FuelLib prediction
predictors = {
    "Density": lambda T: fuel.mixture_density(Y, T),
    "Flash Point": lambda T: fuel.mixture_flash_point(Y, correlation=flash_point_corr),
    "Freezing Point": lambda T: fuel.mixture_freeze_point(Y),
    "Viscosity": lambda T: fuel.mixture_kinematic_viscosity(Y, T, correlation=viscosity_corr),
    "YSI": lambda T: fuel.mixture_yield_sooting_index(Y),
    "Heat of Combustion": lambda T: fuel.mixture_heat_of_combustion(Y),
    "DCN": lambda T: fuel.mixture_derived_cetane_number(Y),
    "Smoke Point (1)": lambda T: fuel.mixture_smoke_point(Y),
    "Smoke Point (2)": lambda T: fuel.mixture_smoke_point(Y),
    "Naphthalenes (2)": lambda T: fuel.mixture_naphthalenes_content(Y),
    "Dist. T10": lambda T: d86().temperature_at(0.10),
    "Dist. T50": lambda T: d86().temperature_at(0.50),
    "Dist. T90": lambda T: d86().temperature_at(0.90),
    "Dist. FBP": lambda T: d86().T_vapor.to("K").max(),
    "Dist. T50 - T10": lambda T: d86().temperature_at(0.50) - d86().temperature_at(0.10),
    "Dist. T90 - T10": lambda T: d86().temperature_at(0.90) - d86().temperature_at(0.10),
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
    if given:
        units = given[0].units
    elif "units" in criteria:
        units = fl.Units.Unit(criteria["units"])  # report-only property
    else:
        raise ValueError(f"No min_value, max_value, value, or units given for {prop}.")

    n = max(
        (
            len(np.atleast_1d(q.magnitude))
            for q in (min_q, max_q, val_q, temp_q)
            if q is not None
        ),
        default=1,
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
results = {
    prop: list(evaluate(prop, criteria)) for prop, criteria in astm_d7566_props.items()
}
statuses = {prop: [row[-1] for row in rows] for prop, rows in results.items()}

# Resolve either/or requirements: hide rows of options that do not apply.
# The first option that is met is the applicable one. If none are met, every
# option is shown so the shortfalls are visible.
hidden = set()
# Requirements that accompany the first entry of an option are indented in the table
indented = {
    p for options in astm_d7566_either_or.values() for option in options for p in option[1:]
}
for options in astm_d7566_either_or.values():
    option_passes = [
        all(s == "PASS" for prop in option for s in statuses[prop])
        for option in options
    ]
    if any(option_passes):
        i = option_passes.index(True)
        hidden |= {p for option in options for p in option} - set(options[i])

print(header)
print("-" * len(header))
for prop, rows in results.items():
    if prop in hidden:
        continue
    label = f"  {prop}" if prop in indented else prop
    for pred, lo, hi, val, T, units, status in rows:
        T_str = "-" if T is None else f"{T.to('celsius').magnitude:.1f} C"
        print(
            f"{label:<18}{_fmt(pred, units):>12}{_fmt(lo, units):>10}"
            f"{_fmt(hi, units):>10}{_fmt(val, units):>10}{T_str:>12}  "
            f"{f'{units:~P}':<12}{status}"
        )


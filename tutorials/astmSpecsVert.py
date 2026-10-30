"""Compare FuelLib predictions and measurements fuel by fuel against ASTM D7566 limits.

Each fuel is spaced out along the horizontal axis and each property has its own
panel (vertical axis). Measurements are black dots, FuelLib predictions are teal
stars, and regions outside the ASTM D7566 specification are shaded red.

Four figures are made, each with its panels stacked vertically: density, viscosity
and heat of combustion; flash point, freezing point, DCN and smoke point; the
distillation temperatures (T10, T50, T90, T100); then the distillation slopes
(T50 - T10 and T90 - T10).

Fuels are taken from every properties file matching ``name_pattern`` that has a
matching gcData file, in both the bundled FuelLib data and the custom fuel data
directory (custom overrides bundled when names collide). Temperature-dependent
properties are compared at the measured temperature nearest the ASTM temperature,
with the prediction made at that same temperature. Fuels without a measurement of a
property are left out of that panel.

Distillation is predicted with the Greenfield et al. D86 simulation, which is slow.
Measured values are interpolated from the measured D86 curve (Temp = D86
temperature, Property_Value = percent recovered). T100 is the final boiling point.
The initial boiling point (T0) is not compared because the Greenfield model does not
capture it.

Notes on the limits:
- DCN, T50 and T90 have no ASTM D7566 limit (report only), so no region is
  shaded.
- Smoke point shades below 18 mm, the lower of the two options (25 mm, or 18 mm with
  naphthalenes <= 3 vol%). Naphthalenes are not measured in the property data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fuellib as fl
from fuellib.data import DEFAULT_DATA_DIR
from fuellib.simulate.distillation import greenfield

# --- User settings ---
custom_dir = Path("/Users/dmontgo2/Documents/FuelLib-FuelData/fuelData")
include_bundled = True
include_custom = True
name_pattern = "posf*"  # glob for fuel names
exclude: set[str] = set()  # fuel names to leave out
viscosity_corr = "Nannoolal"  # "Dutt" or "Nannoolal"
flash_point_corr = "Alibakhshi"  # "Alibakhshi" or "Alqaheem"
max_temp_gap = None  # Max |T_measured - T_target| in K to accept (None = any)
include_distillation = True  # D86 simulation is slow; False skips the distillation figures
fail_alpha = 0.1  # Opacity of the red regions outside the ASTM specification
panel_height = 1.5  # Height in inches of each stacked panel
save_plot = False
display_plot = True
plot_path = Path(__file__).with_name("astm_specs_vert.png")


# --- Measurements ---
def convert(value: float, from_units, to_units: str) -> float:
    """Convert a scalar between units (empty/NaN units are dimensionless)."""
    from_units = "" if pd.isna(from_units) else from_units
    return float(fl.Units.Quantity(float(value), from_units).to(to_units).magnitude)


def lookup(prop: str, units: str, T_target_C: float | None = None):
    """Return a function (fuel) -> (value in units, T in celsius | None) | None.

    For temperature-dependent properties the measurement nearest `T_target_C` is
    used. The function returns None if the fuel has no usable measurement.
    """

    def measure(fuel: fl.Fuel):
        if fuel.propData is None:
            return None
        rows = fuel.propData[fuel.propData["Property"] == prop].copy()
        rows["val"] = pd.to_numeric(rows["Property_Value"], errors="coerce")
        rows = rows.dropna(subset=["val"])
        if rows.empty:
            return None
        if T_target_C is None:
            row = rows.iloc[0]
            return convert(row["val"], row["Property_Units"], units), None

        rows["T_val"] = pd.to_numeric(rows["Temp"], errors="coerce")
        rows = rows.dropna(subset=["T_val"])
        if rows.empty:
            return None
        rows["T_C"] = [
            convert(t, tu, "celsius") for t, tu in zip(rows["T_val"], rows["Temp_Units"])
        ]
        gaps = (rows["T_C"] - T_target_C).abs().to_numpy()
        i = int(np.argmin(gaps))
        if max_temp_gap is not None and gaps[i] > max_temp_gap:
            return None
        row = rows.iloc[i]
        return convert(row["val"], row["Property_Units"], units), float(row["T_C"])

    return measure


# --- Distillation helpers ---
_d86_sim_cache: dict[str, greenfield.D86Result] = {}


def d86_simulated(fuel: fl.Fuel) -> greenfield.D86Result:
    """Simulate the D86 curve of a fuel once and reuse it."""
    if fuel.name not in _d86_sim_cache:
        _d86_sim_cache[fuel.name] = greenfield.distill(fuel, fuel.Y_0)
    return _d86_sim_cache[fuel.name]


def d86_point(fuel: fl.Fuel, fraction: float) -> float | None:
    """Measured D86 temperature in K at a recovered fraction (None if not covered)."""
    if fuel.propData is None:
        return None
    rows = fuel.propData[fuel.propData["Property"] == "Distillation"].copy()
    rows["vol"] = pd.to_numeric(rows["Property_Value"], errors="coerce") / 100.0
    rows["T_val"] = pd.to_numeric(rows["Temp"], errors="coerce")
    rows = rows.dropna(subset=["vol", "T_val"]).sort_values("vol")
    if len(rows) < 2 or not rows["vol"].iloc[0] <= fraction <= rows["vol"].iloc[-1]:
        return None
    T_K = [convert(t, tu, "K") for t, tu in zip(rows["T_val"], rows["Temp_Units"])]
    return float(np.interp(fraction, rows["vol"].to_numpy(), T_K))


def d86_spec(fraction: float, title: str, max_value: float | None = None) -> dict:
    """Spec of the D86 temperature at a recovered fraction (1 = final boiling point)."""

    def measure(fuel: fl.Fuel):
        T = d86_point(fuel, fraction)
        return None if T is None else (convert(T, "K", "celsius"), None)

    def predict(fuel: fl.Fuel, T):
        sim = d86_simulated(fuel)
        if fraction >= 1.0:
            return sim.T_vapor.to("K").max()
        return sim.temperature_at(fraction)

    return {
        "units": "celsius",
        "title": title,
        "ylabel": "[°C]",
        "measure": measure,
        "predict": predict,
        "max_value": max_value,
    }


def d86_span_spec(hi: float, lo: float, title: str, min_value: float | None = None) -> dict:
    """Spec of the D86 temperature difference between two recovered fractions."""

    def measure(fuel: fl.Fuel):
        T_hi, T_lo = d86_point(fuel, hi), d86_point(fuel, lo)
        return None if T_hi is None or T_lo is None else (T_hi - T_lo, None)

    def predict(fuel: fl.Fuel, T):
        sim = d86_simulated(fuel)
        return fl.Units.Quantity(
            float(sim.temperature_at(hi).to("K").magnitude)
            - float(sim.temperature_at(lo).to("K").magnitude),
            "K",
        )

    return {
        "units": "K",
        "title": title,
        "ylabel": "[K]",
        "measure": measure,
        "predict": predict,
        "min_value": min_value,
    }


# --- ASTM D7566 properties ---
#   units:     units used for the comparison
#   title:     panel title
#   ylabel:    vertical axis label (units only)
#   measure:   function (fuel) -> (value in units, T in celsius | None) | None
#   predict:   function (fuel, T) -> Quantity; T is a temperature in K (or None)
#   min_value: (optional) lower ASTM limit in `units`; below it is shaded red
#   max_value: (optional) upper ASTM limit in `units`; above it is shaded red
astm_props = {
    "Density": {
        "units": "kg/m^3",
        "title": "Density at 15 °C",
        "ylabel": "[kg/m$^3$]",
        "measure": lookup("Density", "kg/m^3", 15.0),
        "predict": lambda fuel, T: fuel.mixture_density(fuel.Y_0, T),
        "min_value": 775,
        "max_value": 840,
    },
    "Flash Point": {
        "units": "celsius",
        "title": "Flash Point",
        "ylabel": "[°C]",
        "measure": lookup("FlashPoint", "celsius"),
        "predict": lambda fuel, T: fuel.mixture_flash_point(
            fuel.Y_0, correlation=flash_point_corr
        ),
        "min_value": 38,
    },
    "Freezing Point": {
        "units": "celsius",
        "title": "Freezing Point",
        "ylabel": "[°C]",
        "measure": lookup("FreezePoint", "celsius"),
        "predict": lambda fuel, T: fuel.mixture_freeze_point(fuel.Y_0),
        "max_value": -40,
    },
    "Viscosity -20": {
        "units": "mm^2/s",
        "title": "Viscosity at -20 °C",
        "ylabel": "[mm$^2$/s]",
        "measure": lookup("Viscosity", "mm^2/s", -20.0),
        "predict": lambda fuel, T: fuel.mixture_kinematic_viscosity(
            fuel.Y_0, T, correlation=viscosity_corr
        ),
        "max_value": 8.0,
    },
    "Viscosity -40": {
        "units": "mm^2/s",
        "title": "Viscosity at -40 °C",
        "ylabel": "[mm$^2$/s]",
        "measure": lookup("Viscosity", "mm^2/s", -40.0),
        "predict": lambda fuel, T: fuel.mixture_kinematic_viscosity(
            fuel.Y_0, T, correlation=viscosity_corr
        ),
        "max_value": 12.0,
    },
    "Heat of Combustion": {
        "units": "MJ/kg",
        "title": "Heat of Combustion",
        "ylabel": "[MJ/kg]",
        "measure": lookup("HeatOfCombustion", "MJ/kg"),
        "predict": lambda fuel, T: fuel.mixture_heat_of_combustion(fuel.Y_0),
        "min_value": 42.8,
    },
    "DCN": {
        "units": "dimensionless",
        "title": "Derived Cetane Number",
        "ylabel": "[-]",
        "measure": lookup("DerivedCetaneNumber", "dimensionless"),
        "predict": lambda fuel, T: fuel.mixture_derived_cetane_number(fuel.Y_0),
    },
    "Smoke Point": {
        "units": "mm",
        "title": "Smoke Point",
        "ylabel": "[mm]",
        "measure": lookup("SmokePoint", "mm"),
        "predict": lambda fuel, T: fuel.mixture_smoke_point(fuel.Y_0),
        "min_value": 18,
    },
    "Dist. T10": d86_spec(0.10, "Distillation T10", max_value=205),
    "Dist. T50": d86_spec(0.50, "Distillation T50"),
    "Dist. T90": d86_spec(0.90, "Distillation T90"),
    "Dist. T100": d86_spec(1.00, "Distillation T100 (Final Boiling Point)", max_value=300),
    "Dist. T50 - T10": d86_span_spec(0.50, 0.10, "T50 - T10", min_value=15),
    "Dist. T90 - T10": d86_span_spec(0.90, 0.10, "T90 - T10", min_value=40),
}
if not include_distillation:
    astm_props = {p: s for p, s in astm_props.items() if not p.startswith("Dist.")}


# --- Collect fuels ---
def find_fuels(data_dir: Path) -> list[str]:
    """Return fuel names in data_dir with both property and gcData files."""
    data_dir = Path(data_dir)
    return [
        p.stem
        for p in sorted((data_dir / "propertiesData").glob(f"{name_pattern}.csv"))
        if (data_dir / "gcData" / p.name).exists()
    ]


fuel_dirs: dict[str, Path] = {}
if include_bundled:
    fuel_dirs.update({n: DEFAULT_DATA_DIR for n in find_fuels(DEFAULT_DATA_DIR)})
if include_custom:
    fuel_dirs.update({n: custom_dir for n in find_fuels(custom_dir)})  # overrides
fuel_dirs = {n: d for n, d in sorted(fuel_dirs.items()) if n not in exclude}

# --- Compare ---
rows = []
for name, d in fuel_dirs.items():
    try:
        fuel = fl.Fuel(name, fuelDataDir=None if d == DEFAULT_DATA_DIR else d)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping {name}: {exc}")
        continue
    for prop, spec in astm_props.items():
        meas = spec["measure"](fuel)  # Measure first so unneeded D86 runs are skipped
        if meas is None:
            continue
        value, T_used = meas
        try:
            T = None if T_used is None else fl.Units.Quantity(T_used, "celsius").to("K")
            pred = spec["predict"](fuel, T)
            pred = float(np.squeeze(pred.to(spec["units"]).magnitude))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {name} {prop}: {exc}")
            continue
        rows.append(
            {
                "fuel": name,
                "property": prop,
                "T_used_C": T_used,
                "measured": value,
                "predicted": pred,
            }
        )

data = pd.DataFrame(rows)
if data.empty:
    raise SystemExit("No measured ASTM properties found for the selected fuels.")
data["error"] = data["predicted"] - data["measured"]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(data.to_string(index=False))
print()
print(f"Fuels with data: {data['fuel'].nunique()} of {len(fuel_dirs)}")
for prop, g in data.groupby("property", sort=False):
    print(f"{prop:>18}: N = {len(g):2d}, MAE = {g['error'].abs().mean():.3f}")


# --- Plots ---
def plot_figure(props: list[str], path: Path) -> plt.Figure | None:
    """Stack one panel per property with the fuels spaced along the x axis."""
    sub = data[data["property"].isin(props)]
    if sub.empty:
        return None
    fuels = sorted(sub["fuel"].unique())
    x_of = {f: i for i, f in enumerate(fuels)}

    fig, axes = plt.subplots(
        len(props),
        1,
        sharex=True,
        figsize=(max(8.0, 0.6 * len(fuels) + 3.0), panel_height * len(props) + 1.5),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, prop in zip(axes.ravel(), props):
        spec = astm_props[prop]
        g = sub[sub["property"] == prop]
        ax.set_title(spec["title"], fontsize=14)
        ax.set_ylabel(spec["ylabel"], fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", alpha=0.3)
        if g.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = g["fuel"].map(x_of).to_numpy(dtype=float)
        lo_lim, hi_lim = spec.get("min_value"), spec.get("max_value")
        limits = [v for v in (lo_lim, hi_lim) if v is not None]
        values = np.concatenate([g["measured"], g["predicted"], limits])
        pad = 0.15 * (values.max() - values.min() or 1.0)
        y_lo, y_hi = values.min() - pad, values.max() + pad

        # Regions outside the ASTM specification (limits themselves are in spec)
        if lo_lim is not None:
            ax.axhspan(y_lo, lo_lim, color="red", alpha=fail_alpha, lw=0, zorder=0)
        if hi_lim is not None:
            ax.axhspan(hi_lim, y_hi, color="red", alpha=fail_alpha, lw=0, zorder=0)

        # Both markers sit on the fuel's x position; the dot is drawn on top of the star
        ax.scatter(x, g["predicted"], marker="*", s=220, color="teal",
                   edgecolors="k", linewidths=0.5, zorder=4)
        ax.scatter(x, g["measured"], marker="o", s=50, color="k", zorder=5)
        ax.set_ylim(y_lo, y_hi)  # After the spans, which would otherwise autoscale

    ax = axes.ravel()[-1]
    ax.set_xlim(-0.5, len(fuels) - 0.5)
    ax.set_xticks(range(len(fuels)))
    ax.set_xticklabels([f.removeprefix("posf") for f in fuels], rotation=45, ha="right")
    ax.set_xlabel("Fuel", fontsize=12)

    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="k", linestyle="", markersize=8, label="Data"),
            Line2D([], [], marker="*", color="teal", markeredgecolor="k", linestyle="",
                   markersize=14, label="FuelLib"),
            Patch(facecolor="red", alpha=fail_alpha, label="Outside ASTM D7566"),
        ],
        loc="outside lower center",
        ncol=3,
        fontsize=12,
    )
    if save_plot:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    return fig


# Figures: file suffix -> properties. Properties not in `astm_props` (e.g. distillation
# when it is disabled) are dropped, and figures left empty are skipped.
figure_groups = {
    "1": ["Density", "Viscosity -20", "Viscosity -40", "Heat of Combustion"],
    "2": ["Flash Point", "Freezing Point", "DCN", "Smoke Point"],
    "distillation": ["Dist. T10", "Dist. T50", "Dist. T90", "Dist. T100"],
    "distillation_slopes": ["Dist. T50 - T10", "Dist. T90 - T10"],
}
for suffix, group in figure_groups.items():
    plot_figure(
        [p for p in group if p in astm_props],
        plot_path.with_name(f"{plot_path.stem}_{suffix}{plot_path.suffix}"),
    )
if display_plot:
    plt.show()

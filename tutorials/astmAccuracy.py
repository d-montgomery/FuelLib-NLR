"""Compare FuelLib predictions to measurements for the ASTM D7566 properties.

Fuels are taken from every properties file matching ``name_pattern`` (default
``posf*``) that has a matching gcData file, in both the bundled FuelLib data and the
custom fuel data directory (custom overrides bundled when names collide).

For each fuel and ASTM property, the measured value is read from the fuel's property
data. Temperature-dependent properties (density, viscosity) are evaluated at the ASTM
temperatures; if no measurement exists at that temperature, the nearest measured
temperature is used and the prediction is made at that same temperature so the
comparison is like for like. Properties without a measurement are skipped.

Distillation (T10, T50, T90, final boiling point and the T50 - T10 and T90 - T10
spans) is predicted with the Greenfield et al. D86 simulation. The measured values
are interpolated from the fuel's measured D86 curve (Temp = D86 temperature,
Property_Value = percent recovered); the final boiling point is the temperature at
100 % recovered.

YSI and naphthalenes content are not compared because they are not measured in the
property data.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Circle

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
band_tol = 0.10  # Default relative tolerance band (±) on the parity plots
include_distillation = True  # D86 simulation is slow; False skips the distillation figure
save_plot = False
display_plot = True
plot_path = Path(__file__).with_name("astm_accuracy.png")

_d86_sim_cache: dict[str, greenfield.D86Result] = {}


def d86_simulated(fuel: fl.Fuel) -> greenfield.D86Result:
    """Simulate the D86 curve of a fuel once and reuse it."""
    if fuel.name not in _d86_sim_cache:
        _d86_sim_cache[fuel.name] = greenfield.distill(fuel, fuel.Y_0)
    return _d86_sim_cache[fuel.name]


def d86_measured(fuel: fl.Fuel) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (fraction recovered, D86 temperature in K) of the measured curve."""
    if fuel.propData is None:
        return None
    rows = fuel.propData[fuel.propData["Property"] == "Distillation"].copy()
    rows["vol"] = pd.to_numeric(rows["Property_Value"], errors="coerce") / 100.0
    rows["T_val"] = pd.to_numeric(rows["Temp"], errors="coerce")
    rows = rows.dropna(subset=["vol", "T_val"]).sort_values("vol")
    if len(rows) < 2:
        return None
    T_K = [_convert(t, tu, "K") for t, tu in zip(rows["T_val"], rows["Temp_Units"])]
    return rows["vol"].to_numpy(), np.array(T_K)


def d86_point(fuel: fl.Fuel, fraction: float) -> float | None:
    """Measured D86 temperature in K at a recovered fraction (None if not covered)."""
    curve = d86_measured(fuel)
    if curve is None or not curve[0][0] <= fraction <= curve[0][-1]:
        return None
    return float(np.interp(fraction, *curve))


def d86_span(fuel: fl.Fuel, hi: float, lo: float, measured: bool) -> float | None:
    """Difference in K between the D86 temperatures at fractions `hi` and `lo`."""
    if measured:
        T_hi, T_lo = d86_point(fuel, hi), d86_point(fuel, lo)
        return None if T_hi is None or T_lo is None else T_hi - T_lo
    sim = d86_simulated(fuel)
    return float(sim.temperature_at(hi).magnitude - sim.temperature_at(lo).magnitude)


def d86_property(fraction: float, label: str) -> dict:
    """Spec of the D86 temperature at a recovered fraction (1.0 = final boiling point)."""
    return {
        "units": "celsius",
        "label": label,
        "temps_C": None,
        "measure": lambda fuel: (
            None
            if (T := d86_point(fuel, fraction)) is None
            else _convert(T, "K", "celsius")
        ),
        "predict": lambda fuel, T: (
            d86_simulated(fuel).T_vapor.max()
            if fraction >= 1.0
            else d86_simulated(fuel).temperature_at(fraction)
        ),
        "band_tol": 0.05,
    }


def d86_span_property(hi: float, lo: float, label: str) -> dict:
    """Spec of the D86 temperature difference between two recovered fractions."""
    return {
        "units": "K",
        "label": label,
        "temps_C": None,
        "measure": lambda fuel: d86_span(fuel, hi, lo, measured=True),
        "predict": lambda fuel, T: fl.Units.Quantity(
            d86_span(fuel, hi, lo, measured=False), "K"
        ),
    }


# ASTM D7566 properties compared against measurements.
#   data:    "Property" name in the fuel's property data
#   units:   units used for the comparison
#   temps_C: target temperatures in celsius (None if not temperature dependent)
#   predict: function (fuel, T) -> Quantity; T is a temperature in K (or None)
#   band_tol: (optional) relative tolerance band (±) overriding `band_tol` above
#   measure: (optional) function (fuel) -> value in `units` or None, replacing the
#            default lookup of `data` in the fuel's property data
astm_props = {
    "Density": {
        "data": "Density",
        "units": "kg/m^3",
        "label": "Density [kg/m$^3$]",
        "temps_C": [15.0],
        "predict": lambda fuel, T: fuel.mixture_density(fuel.Y_0, T),
        "band_tol": 0.01,  # Densities span only ~10%, so 10% would be meaningless
    },
    "Flash Point": {
        "data": "FlashPoint",
        "units": "celsius",
        "label": "Flash point [°C]",
        "temps_C": None,
        "predict": lambda fuel, T: fuel.mixture_flash_point(
            fuel.Y_0, correlation=flash_point_corr
        ),
    },
    "Freezing Point": {
        "data": "FreezePoint",
        "units": "celsius",
        "label": "Freezing point [°C]",
        "temps_C": None,
        "predict": lambda fuel, T: fuel.mixture_freeze_point(fuel.Y_0),
    },
    "Viscosity": {
        "data": "Viscosity",
        "units": "mm^2/s",
        "label": "Kinematic viscosity [mm$^2$/s]",
        "temps_C": [-20.0, -40.0],
        "predict": lambda fuel, T: fuel.mixture_kinematic_viscosity(
            fuel.Y_0, T, correlation=viscosity_corr
        ),
    },
    "Heat of Combustion": {
        "data": "HeatOfCombustion",
        "units": "MJ/kg",
        "label": "Heat of combustion [MJ/kg]",
        "temps_C": None,
        "predict": lambda fuel, T: fuel.mixture_heat_of_combustion(fuel.Y_0),
        "band_tol": 0.01,
    },
    "DCN": {
        "data": "DerivedCetaneNumber",
        "units": "dimensionless",
        "label": "Derived cetane number [-]",
        "temps_C": None,
        "predict": lambda fuel, T: fuel.mixture_derived_cetane_number(fuel.Y_0),
    },
    "Smoke Point": {
        "data": "SmokePoint",
        "units": "mm",
        "label": "Smoke point [mm]",
        "temps_C": None,
        "predict": lambda fuel, T: fuel.mixture_smoke_point(fuel.Y_0),
    },
    "Dist. T10": d86_property(0.10, "Distillation T10 [°C]"),
    "Dist. T50": d86_property(0.50, "Distillation T50 [°C]"),
    "Dist. T90": d86_property(0.90, "Distillation T90 [°C]"),
    "Dist. FBP": d86_property(1.00, "Final boiling point [°C]"),
    "Dist. T50 - T10": d86_span_property(0.50, 0.10, "T50 - T10 [K]"),
    "Dist. T90 - T10": d86_span_property(0.90, 0.10, "T90 - T10 [K]"),
}
if not include_distillation:
    astm_props = {p: s for p, s in astm_props.items() if not p.startswith("Dist.")}


def find_fuels(data_dir: Path) -> list[str]:
    """Return fuel names in data_dir with both property and gcData files."""
    data_dir = Path(data_dir)
    return [
        p.stem
        for p in sorted((data_dir / "propertiesData").glob(f"{name_pattern}.csv"))
        if (data_dir / "gcData" / p.name).exists()
    ]


def _convert(value: float, from_units, to_units: str) -> float:
    """Convert a scalar between units (empty/NaN units are dimensionless)."""
    from_units = "" if pd.isna(from_units) else from_units
    return float(fl.Units.Quantity(float(value), from_units).to(to_units).magnitude)


def measured_value(
    fuel: fl.Fuel, prop: str, units: str, T_target_C: float | None
) -> tuple[float, float | None] | None:
    """Return (value in units, temperature in celsius) of the best measurement.

    For temperature-dependent properties the measurement nearest to `T_target_C` is
    chosen. Returns None if there is no usable measurement.
    """
    df = fuel.propData
    if df is None:
        return None
    rows = df[df["Property"] == prop].copy()
    rows["val"] = pd.to_numeric(rows["Property_Value"], errors="coerce")
    rows = rows.dropna(subset=["val"])
    if T_target_C is None:
        if rows.empty:
            return None
        row = rows.iloc[0]
        return _convert(row["val"], row["Property_Units"], units), None

    rows["T_val"] = pd.to_numeric(rows["Temp"], errors="coerce")
    rows = rows.dropna(subset=["T_val"])
    if rows.empty:
        return None
    rows["T_C"] = [
        _convert(t, tu, "celsius") for t, tu in zip(rows["T_val"], rows["Temp_Units"])
    ]
    gaps = (rows["T_C"] - T_target_C).abs().to_numpy()
    i = int(np.argmin(gaps))
    if max_temp_gap is not None and gaps[i] > max_temp_gap:
        return None
    row = rows.iloc[i]
    return _convert(row["val"], row["Property_Units"], units), float(row["T_C"])


# --- Collect fuels ---
fuel_dirs: dict[str, Path] = {}
if include_bundled:
    fuel_dirs.update({n: DEFAULT_DATA_DIR for n in find_fuels(DEFAULT_DATA_DIR)})
if include_custom:
    fuel_dirs.update({n: custom_dir for n in find_fuels(custom_dir)})  # overrides
fuel_dirs = {n: d for n, d in sorted(fuel_dirs.items()) if n not in exclude}

# --- Compare ---
rows = []
meta_names: dict[str, str] = {}  # fuel -> display name from fuel_metadata.yaml
for name, d in fuel_dirs.items():
    try:
        fuel = fl.Fuel(name, fuelDataDir=None if d == DEFAULT_DATA_DIR else d)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping {name}: {exc}")
        continue
    meta_names[name] = (fuel.metadata or {}).get("name", name)
    for prop, spec in astm_props.items():
        for T_target in spec["temps_C"] or [None]:
            if "measure" in spec:
                value = spec["measure"](fuel)
                meas = None if value is None else (value, None)
            else:
                meas = measured_value(fuel, spec["data"], spec["units"], T_target)
            if meas is None:
                continue
            value, T_used = meas
            try:
                T = None if T_used is None else fl.Units.Quantity(T_used, "celsius")
                T = None if T is None else T.to("K")
                pred = spec["predict"](fuel, T)
                pred = float(np.squeeze(pred.to(spec["units"]).magnitude))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {name} {prop}: {exc}")
                continue
            rows.append(
                {
                    "fuel": name,
                    "property": prop,
                    "T_target_C": T_target,
                    "T_used_C": T_used,
                    "measured": value,
                    "predicted": pred,
                }
            )

data = pd.DataFrame(rows)
if data.empty:
    raise SystemExit("No measured ASTM properties found for the selected fuels.")
data["error"] = data["predicted"] - data["measured"]
data["error_%"] = 100.0 * data["error"] / data["measured"]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(data.to_string(index=False))
print()
print(f"Fuels with data: {data['fuel'].nunique()} of {len(fuel_dirs)}")
for (prop, T_target), g in data.groupby(["property", "T_target_C"], dropna=False):
    T_str = "" if pd.isna(T_target) else f" at {T_target:.0f} C"
    print(
        f"{prop + T_str:>26}: N = {len(g):2d}, "
        f"MAE = {g['error'].abs().mean():.3f}, bias = {g['error'].mean():+.3f}, "
        f"MAPE = {g['error_%'].abs().mean():.1f}%"
    )

# --- Plots ---
# Style matches fuellib.cli.plotting
# Color encodes the fuel; marker shape encodes the temperature (when a property has
# more than one).
fuel_names = sorted(data["fuel"].unique())
# Fuel categories are named "<letter>-<n> ..." in the fuel metadata. Cat A fuels are
# orange/red and Cat C fuels are blue/purple/green; all other fuels use the remaining
# colors (browns, pinks, grays, olives) so they cannot be mistaken for A or C.
cat_a = [n for n in fuel_names if re.match(r"A-\d", meta_names.get(n, ""))]
cat_c = [n for n in fuel_names if re.match(r"C-\d", meta_names.get(n, ""))]
others = [n for n in fuel_names if n not in cat_a + cat_c]
cat_a_palette = ["#EDC948", "tab:orange", "tab:red"]  # Tableau yellow, orange, red
cat_c_palette = [
    "tab:blue",
    "tab:purple",
    "tab:green",
    "tab:cyan",
    "#1F3A93",  # Navy
    "#D81B9B",  # Magenta
    "#00897B",  # Teal
    "#7CB342",  # Light green
]
tab20 = plt.get_cmap("tab20")
other_palette = [tab20(i) for i in range(10, 18)]  # browns, pinks, grays, olives
fuel_colors = {}
for group, palette in (
    (cat_a, cat_a_palette),
    (cat_c, cat_c_palette),
    (others, other_palette),
):
    fuel_colors.update({n: palette[i % len(palette)] for i, n in enumerate(group)})
fuel_names = cat_a + cat_c + others  # Legend order: Cat A, Cat C, then the rest
markers = ["o", "s", "^", "D"]
ncols = 4


def draw_panel(ax, prop: str, spec: dict) -> None:
    """Draw the measured vs. predicted parity plot of one property on `ax`."""
    sub = data[data["property"] == prop]
    ax.set_title(prop, fontsize=16)
    ax.set_xlabel(f"Measured\n{spec['label']}", fontsize=12)
    ax.set_ylabel(f"Predicted\n{spec['label']}", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    both = np.concatenate([sub["measured"], sub["predicted"]])
    pad = 0.08 * (both.max() - both.min() or 1.0)
    lo, hi = both.min() - pad, both.max() + pad
    xx = np.array([lo, hi])
    tol = spec.get("band_tol", band_tol)
    ax.plot(xx, xx, "k-", linewidth=2, label="Ideal")
    ax.fill_between(
        xx,
        xx - tol * np.abs(xx),
        xx + tol * np.abs(xx),
        color="0.9",
        label=f"±{tol:.0%}",
    )

    series = sub.groupby("T_target_C", dropna=False, sort=False)
    for k, (T_target, g) in enumerate(series):
        is_open = [n in cat_c for n in g["fuel"]]  # Cat C fuels use open markers
        ax.scatter(
            g["measured"],
            g["predicted"],
            marker=markers[k % len(markers)],
            facecolors=[
                "none" if o else fuel_colors[n] for o, n in zip(is_open, g["fuel"])
            ],
            edgecolors=[
                fuel_colors[n] if o else "k" for o, n in zip(is_open, g["fuel"])
            ],
            linewidths=[2 if o else 1 for o in is_open],
            s=75,
            zorder=5,
        )
        if pd.notna(T_target) and sub["T_target_C"].nunique() > 1:
            # Gray proxy marker so the legend explains the marker shape
            ax.scatter(
                [],
                [],
                marker=markers[k % len(markers)],
                c="0.6",
                edgecolors="k",
                linewidths=1,
                s=75,
                label=f"{T_target:.0f} °C",
            )
    ax.text(
        0.97,
        0.03,
        f"N = {len(sub)}\nMAE = {sub['error'].abs().mean():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)

# Fuel legend: built from offset boxes (rather than with `legend`) so the group
# headers are left aligned while the fuel entries are indented beneath them.
groups = [
    ("Category A", cat_a),
    ("Category C", cat_c),
    ("Other", others),
]
groups = [(title, group) for title, group in groups if group]
legend_fontsize = 16
n_rows = 1 + sum(len(group) + 1 for _, group in groups)
# Shrink the font if there are too many rows to fit in the cell (~300 pt tall)
legend_fontsize = min(legend_fontsize, 300 / (n_rows * 1.9))
fs = legend_fontsize


def _entry(name: str) -> HPacker:
    """Marker + label row for one fuel, indented under its group header."""
    is_open = name in cat_c
    marker = DrawingArea(fs * 1.2, fs * 1.2, 0, 0)
    marker.add_artist(
        Circle(
            (fs * 0.6, fs * 0.6),
            fs * 0.4,
            facecolor="none" if is_open else fuel_colors[name],
            edgecolor=fuel_colors[name] if is_open else "k",
            linewidth=2 if is_open else 1,
        )
    )
    indent = DrawingArea(fs * 1.0, 1, 0, 0)
    label = TextArea(name.removeprefix("posf"), textprops={"fontsize": fs})
    return HPacker(children=[indent, marker, label], align="center", pad=0, sep=fs * 0.4)


def draw_fuel_legend(ax) -> None:
    """Draw the fuel legend centered in the (otherwise empty) axes `ax`."""
    ax.set_visible(True)
    ax.axis("off")
    blocks = [TextArea("NJFCP Fuels", textprops={"fontsize": fs, "fontweight": "bold"})]
    for title, group in groups:
        header = TextArea(title, textprops={"fontsize": fs})
        entries = [_entry(n) for n in group]
        blocks.append(
            VPacker(children=[header, *entries], align="left", pad=0, sep=fs * 0.35)
        )
    legend_box = VPacker(children=blocks, align="left", pad=0, sep=fs * 0.8)
    # Anchored at the center of the cell, so it stays centered when the window resizes
    ax.add_artist(
        AnchoredOffsetbox(
            loc="center", child=legend_box, pad=0, borderpad=0, frameon=False
        )
    )


# One figure for the ASTM D7566 properties and one for distillation. Each has one
# panel per property and the fuel legend in the last cell of the grid.
figures = {
    "ASTM D7566 properties": [p for p in astm_props if not p.startswith("Dist.")],
    "Distillation": [p for p in astm_props if p.startswith("Dist.")],
}
for k, (fig_title, prop_names) in enumerate(figures.items()):
    if not prop_names:
        continue
    nrows = int(np.ceil((len(prop_names) + 1) / ncols))
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True
    )
    axes = axes_grid.ravel()
    for ax, prop in zip(axes, prop_names):
        draw_panel(ax, prop, astm_props[prop])
    for ax in axes[len(prop_names) : -1]:
        ax.set_visible(False)
    draw_fuel_legend(axes[-1])
    fig.suptitle(fig_title, fontsize=18, fontweight="bold")

    if save_plot:
        path = plot_path if k == 0 else plot_path.with_name(
            f"{plot_path.stem}_{fig_title.lower().replace(' ', '_')}{plot_path.suffix}"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
if display_plot:
    plt.show()

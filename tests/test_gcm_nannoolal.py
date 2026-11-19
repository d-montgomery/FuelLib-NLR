"""Test the Nannoolal decomposition against the bundled reference data."""

import numpy as np
import pandas as pd
import pytest

from fuellib import DEFAULT_DATA_DIR, Fuel, Units
from fuellib.correlate import components, mixture
from fuellib.gcm.nannoolal import (
    TABLE,
    _contribution_sum,
    _group_interaction,
    decompose,
)

REF_COMPOUNDS = pd.read_csv(DEFAULT_DATA_DIR / "ref_compounds.csv").set_index(
    "Reference Compound"
)
REF_NANNOOLAL = (
    pd
    .read_csv(DEFAULT_DATA_DIR / "ref_nannoolal.csv")
    .set_index("Reference Compound")
    .fillna(0)
    .astype(int)
)


def test_reference_names_match_ref_compounds():
    """Every ref_nannoolal compound should be named as in ref_compounds."""
    missing = REF_NANNOOLAL.index.difference(REF_COMPOUNDS.index)
    assert missing.empty, f"Not in ref_compounds.csv: {list(missing)}"


def test_reference_groups_match_gcm_table():
    """Every ref_nannoolal group column should be a row of the GCM table."""
    missing = REF_NANNOOLAL.columns.difference(TABLE.index)
    assert missing.empty, f"Not in nannoolal.csv: {list(missing)}"


@pytest.mark.parametrize(
    "name",
    REF_NANNOOLAL.index.intersection(REF_COMPOUNDS.index),
)
def test_reference_decomposition(name: str):
    """The SMILES decomposition should match the tabulated Nannoolal groups."""
    smiles = REF_COMPOUNDS.loc[name, "SMILES"]
    computed = decompose(smiles).iloc[0]
    expected = REF_NANNOOLAL.loc[name].reindex(TABLE.index, fill_value=0)

    diff = {
        group: (int(computed[group]), int(expected[group]))
        for group in TABLE.index
        if computed[group] != expected[group]
    }
    assert not diff, f"{name} ({smiles}) group (computed, tabulated): {diff}"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CCO", {"CH3-(ne)": 1, ">C(c)<(e)": 1, "-OH (<C5)": 1}),
        (
            "CCCCCO",
            {"CH3-(ne)": 1, ">C(c)<(e)": 1, "-C(c)H2-": 3, "-OH (>C4)": 1},
        ),
        ("CC(C)O", {"CH3-(ne)": 2, ">C(c)<(e)": 1, "-OH (sec)": 1}),
        ("CC(C)(C)O", {"CH3-(ne)": 3, ">C(c)<(e)": 1, "-OH (tert)": 1}),
        ("CCCCO", {"CH3-(ne)": 1, ">C(c)<(e)": 1, "-C(c)H2-": 2, "-OH (<C5)": 1}),
        (
            "OC1CCCCC1",
            {">C(r)<(e,c)": 1, "-C(r)H2-": 5, "-OH (sec)": 1},
        ),
        (
            "Oc1ccccc1",
            {"=C(a)H-": 5, "=C(a)<(e)": 1, "HO-(C(a))": 1},
        ),
        (
            "OCc1ccccc1",
            {"=C(a)H-": 5, ">C(c)<(e)": 1, "=C(a)<(ne)": 1, "-OH (>C4)": 1},
        ),
        (
            "OCCO",
            {">C(c)<(e)": 2, "-OH (<C5)": 2},
        ),
        (
            "Oc1ccc(O)cc1",
            {"=C(a)H-": 4, "=C(a)<(e)": 2, "HO-(C(a))": 2, "Para-pair(s)": 1},
        ),
    ],
)
def test_alcohol_decomposition(smiles: str, expected: dict[str, int]):
    """Alcohols should be assigned to the Nannoolal alkanol/phenol OH groups."""
    computed = decompose(smiles).iloc[0]
    assert computed[computed != 0].to_dict() == expected


def test_viscosity_group_interaction():
    """GI of dBv and Tv uses f_GI = 1 and 2 (Nannoolal 2009 Eq. 10) for diols."""
    decomp = decompose("OCCO")
    n = np.array([4.0])
    dBv = _contribution_sum(decomp, "dBv", n, 1.0)[0]
    Tv = _contribution_sum(decomp, "Tv", n, 2.0)[0]
    assert dBv == pytest.approx(2 * 0.2188 + 2 * 0.0213473 + 2 * -0.1124939 / 4)
    assert Tv == pytest.approx(
        2 * 1284.745 + 2 * 103.4109 + 2 * (2 * -1313.569) / 4
    )


def test_group_interaction_three_groups():
    """GI should scale as 1/(m - 1) with m the number of non-additive groups."""
    decomp = decompose("OCC(O)CO")  # glycerol: three OH, three pairs
    n = np.array([6.0])
    gi = _group_interaction(decomp, "dBv", n, 1.0)[0]
    assert gi == pytest.approx(2 * 3 * -0.1124939 / (6 * 2))


def test_group_interaction_single_group():
    """No interaction should be applied for a single non-additive group."""
    gi = _group_interaction(decompose("CCCCCO"), "dBv", np.array([6.0]), 1.0)
    assert gi[0] == 0.0


@pytest.mark.parametrize("fuel_name", ["heptane", "posf10325"])
def test_viscosity(fuel_name: str):
    """Nannoolal viscosities should be finite and positive with correct units."""
    fuel = Fuel(fuel_name)
    T = Units.Quantity(298.15, "K")

    mu_i = components.viscosity_dynamic_nannoolal(fuel, T)

    nu_i = components.viscosity_kinematic_nannoolal(fuel, T)
    for prop, units in [(mu_i, "Pa*s"), (nu_i, "m^2/s")]:
        assert prop.check(units)
        assert prop.magnitude.shape == (fuel.num_compounds,)
        assert np.all(np.isfinite(prop.magnitude))
        assert np.all(prop.magnitude > 0)

    mu = mixture.viscosity_dynamic_nannoolal(fuel, T)
    nu = mixture.viscosity_kinematic_nannoolal(fuel, T)
    for prop, units in [(mu, "Pa*s"), (nu, "m^2/s")]:
        assert prop.check(units)
        assert np.ndim(prop.magnitude) == 0
        assert np.isfinite(prop.magnitude)
        assert prop.magnitude > 0

"""Nannoolal group contribution parameters."""

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
from rdkit import Chem

from ..utils import Units, types
from .core import Decomposer, GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

_ALL_ROWS = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
TABLE = _ALL_ROWS[_ALL_ROWS["Order"] < 3]
"""First- and second-order group contributions (rows of the decomposition)."""
INTERACTIONS = _ALL_ROWS[_ALL_ROWS["Order"] == 3]
"""Group interaction contributions (counted from the first-order groups)."""
nannoolal_gcm = GCMRegistry.register("nannoolal", property_fns=[])

# Non-additive group families: family -> first-order groups that belong to it
NON_ADDITIVE_GROUPS = {
    "OH": ("-OH (<C5)", "-OH (>C4)", "-OH (sec)"),
    "OH(a)": ("HO-(C(a))",),
}
# Interaction group name (row of INTERACTIONS) -> interacting non-additive families
INTERACTION_PAIRS = {
    "OH-OH": ("OH", "OH"),
    "OH(a)-OH(a)": ("OH(a)", "OH(a)"),
}


def _get_row(property_name: str) -> types.Array1D:
    """Get property row from GCM table.

    Args:
        property_name: Name of the property to retrieve from the GCM table.

    Returns:
        The property row as a 1D numpy array.

    Raises:
        KeyError: If the property is not found in the GCM table.
    """
    if property_name not in TABLE.columns:
        msg = f"Property '{property_name}' not found in GCM table."
        raise KeyError(msg)
    return TABLE[property_name].to_numpy(dtype=float)


def _heavy_atom_counts(fuel: "Fuel") -> types.Array1D:
    """Count the non-hydrogen atoms of each component of the fuel.

    Args:
        fuel: The fuel object containing the component molecules.

    Returns:
        The number of non-hydrogen atoms (n) of each component.
    """
    return np.array([m.GetNumHeavyAtoms() for m in fuel.comps.rdkit_mols], dtype=float)


def _group_interaction(
    decomp: pd.DataFrame, property_name: str, n: types.Array1D, f_gi: float
) -> types.Array1D:
    """Compute the group interaction contribution GI (Nannoolal Eq. 3 and Eq. 10).

    Args:
        decomp: Group decomposition (components in rows, groups in columns).
        property_name: Name of the column of INTERACTIONS to use.
        n: Number of non-hydrogen atoms of each component.
        f_gi: Scaling factor (2 for Tv, 1 for dBv).

    Returns:
        The group interaction contribution of each component. It is zero for
        components with fewer than two non-additive groups.
    """
    counts = decomp.reindex(columns=TABLE.index, fill_value=0)
    families = {
        family: counts.reindex(columns=list(groups), fill_value=0)
        .sum(axis=1)
        .to_numpy(dtype=float)
        for family, groups in NON_ADDITIVE_GROUPS.items()
    }
    m = sum(families.values())

    pair_sum = np.zeros_like(m)
    for name, (fam_i, fam_j) in INTERACTION_PAIRS.items():
        k_i, k_j = families[fam_i], families[fam_j]
        pairs = k_i * (k_i - 1) / 2 if fam_i == fam_j else k_i * k_j
        pair_sum += np.where(
            pairs != 0, pairs * INTERACTIONS.loc[name, property_name], 0.0
        )

    # Each interaction appears twice in the double sum of Eq. 3 (C_i-j = C_j-i)
    denom = np.where(m > 1, m - 1, 1.0)
    return np.where(m > 1, f_gi / n * 2 * pair_sum / denom, 0.0)


def _contribution_sum(
    decomp: pd.DataFrame, property_name: str, n: types.Array1D, f_gi: float
) -> types.Array1D:
    """Sum the group contributions and the group interaction contribution.

    Args:
        decomp: Group decomposition (components in rows, groups in columns).
        property_name: Name of the property column in the GCM table.
        n: Number of non-hydrogen atoms of each component.
        f_gi: Scaling factor of the group interaction contribution.

    Returns:
        The sum of the group contributions plus GI for each component.
    """
    nij = decomp.reindex(columns=TABLE.index, fill_value=0).to_numpy()
    xj = _get_row(property_name)
    _check_compatible_dims(xj, nij)
    # Only absent groups are zeroed; NaN propagates for used, unsupported groups
    xi = np.where(nij != 0, nij * xj, 0.0).sum(axis=1)
    return xi + _group_interaction(decomp, property_name, n, f_gi)


def _check_compatible_dims(
    xj: Annotated[types.Array1D, "GCM property vector"],
    Aij: Annotated[types.Array2D, "Group decomposition matrix"],
) -> None:
    """Check if the shape of the GCM properties and group decomposition matrix match.

    Args:
        xj: The GCM property vector.
        Aij: The group decomposition matrix.

    Raises:
        ValueError: If the dimensions of the GCM contribution vector and the group
            decomposition matrix are incompatible.
    """
    if Aij.shape[1] != xj.shape[0]:
        msg = (
            f"Incompatible dimensions: GCM contribution vector has length {xj.shape[0]}"
            f", but group decomposition matrix has {Aij.shape[1]} columns."
        )
        raise ValueError(msg)


# Connected to N, O, F or Cl (aliphatic or aromatic N/O)
E = "$(*~[#7,#8,F,Cl])"

# (priority, Nannoolal group name, SMARTS); lower priority number = matched first
order1_groups = [
    (5, ">C=C=C<", "[C]=[C]=[C]"),
    (6, ">C=C-C=C< (ring)", "[C]=@[C]-[C]=@[C]"),
    (7, ">C=C-C=C< (chain)", "[C]=!@[C]-[C]=!@[C]"),
    (8, "-C#C-C#C-", "[C]#[C]-[C]#[C]"),
    (56, "HC#C-", "[CH1]#[C]"),
    (57, "H2C(c)=C<", "[CH2]=[C]"),
    (58, "-(e)C(c)=C(c)<", f"[C;{E}]=!@[C]"),
    (59, ">C(c)=C(c)<(C(a))", "[C;$(*~c)]=!@[C]"),
    (60, ">C(r)=C(r)<", "[C]=@[C]"),
    (61, "-C#C-", "[CH0]#[CH0]"),
    (62, ">C(c)=C(c)<", "[C;!H2]=!@[C;!H2]"),
    # Alcohols: only the O atom is matched; the carbinol carbon is matched separately.
    # Primary alcohols are "-OH (>C4)" and are reassigned to "-OH (<C5)" in
    # `NannoolalDecomposer.first_order_counts` if the molecule has fewer than five C.
    (88, "-OH (>C4)", "[OX2H1;$(O[CX4;H2,H3])]"),
    (89, "HO-(C(a))", "[OX2H1;$(O[c])]"),
    (90, "-OH (sec)", "[OX2H1;$(O[CX4H1]([#6])[#6])]"),
    (91, "-OH (tert)", "[OX2H1;$(O[CX4D4])]"),
    (103, "CH3-(e)", f"[CH3;{E}]"),
    (104, "CH3-(a)", "[CH3;$(*~a)]"),
    (105, "CH3-(ne)", f"[CH3;!{E}]"),
    (106, "=C(a)H-", "[cH1]"),
    (107, ">C(r)<(Ca)", "[CX4;R;$(*~c)]"),
    (108, ">C(c)<(e)", f"[CX4;!R;!H3;{E}]"),
    (109, ">C(c)<(a)", "[CX4;!R;!H3;$(*~c)]"),
    (110, ">C(r)<(e,c)", "[CX4;R;$(*!@[#7,#8]),$(*~[F,Cl])]"),
    (111, ">C(r)<(e,r)", "[CX4;R;$(*@[#7,#8])]"),
    (112, "-C(c)H2-", "[CH2X4;!R]"),
    (113, "-C(r)H2-", "[CH2X4;R]"),
    (114, "=C(a)<(e)", f"[cH0;{E}]"),
    (115, "(a)=C(a)<2(a)", "[cH0;$(*(:a)(:a):a)]"),
    (116, "C(a)=C(a)<C2(a) (bridge)", "[cH0;$(*(:a)(:a)!:a)]"),
    (117, "=C(a)<(ne)", f"[cH0;!{E}]"),
    (118, ">C(r)H-", "[CH1X4;R]"),
    (119, ">C(c)H-", "[CH1X4;!R]"),
    (120, ">C(r)<", "[CH0X4;R]"),
    (121, ">C(c)<", "[CH0X4;!R]"),
]
# Carbon whose heavy-atom neighbors are all carbon
CC = "!$(*~[!#6;!#1])"

# (Nannoolal group name, SMARTS); additive, so every unique match is counted
order2_groups = [
    ("((C=)(C)C-CC3)", f"[#6X3;D3;{CC};$(*=,:[#6])]-[CX4;D4;{CC}]"),
    ("C2C-CC2", f"[CX4;D3;{CC}]-[CX4;D3;{CC}]"),
    ("C3C-CC2", f"[CX4;D4;{CC}]-[CX4;D3;{CC}]"),
    ("C3C-CC3", f"[CX4;D4;{CC}]-[CX4;D4;{CC}]"),
]

# Ring position distance between two substituents on a benzene ring
PAIR_GROUPS = {1: "Ortho-pair(s)", 2: "Meta-pair(s)", 3: "Para-pair(s)"}


class NannoolalDecomposer(Decomposer):
    """Decomposer for first and second order Nannoolal groups."""

    def first_order_counts(self, mol: Chem.Mol, smiles: str) -> dict[str, int]:
        """Count first order groups, assigning short-chain primary alcohols.

        Primary alcohols are matched as "-OH (>C4)" and are reassigned to "-OH (<C5)"
        if the molecule has fewer than five carbons.

        Args:
            mol: The parsed molecule.
            smiles: The SMILES string being decomposed, used in error messages.

        Returns:
            A mapping of first order group name to its number of occurrences.
        """
        counts = super().first_order_counts(mol, smiles)
        n_carbons = sum(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
        if n_carbons < 5 and "-OH (>C4)" in counts:
            counts["-OH (<C5)"] = counts.pop("-OH (>C4)")
        return counts

    def second_order_counts(self, mol: Chem.Mol, smiles: str) -> dict[str, int]:
        """Count second order groups, hydrogen counts, rings, and aromatic pairs.

        Args:
            mol: The parsed molecule.
            smiles: The SMILES string being decomposed, used in error messages.

        Returns:
            A mapping of second order group name to its number of occurrences.
        """
        counts: dict[str, int] = {}

        for group, smarts in order2_groups:
            query = self.parse_smarts(smiles, smarts)
            if n_matches := len(mol.GetSubstructMatches(query)):
                counts[group] = n_matches

        n_hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        if n_hydrogens == 0:
            counts["No hydrogen"] = 1
        elif n_hydrogens == 1:
            counts["One hydrogen"] = 1

        # Count each non-aromatic ring once (rings may share atoms in fused systems)
        ring_info = mol.GetRingInfo()
        for atom_ring, bond_ring in zip(
            ring_info.AtomRings(), ring_info.BondRings(), strict=True
        ):
            if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring):
                continue
            if len(atom_ring) in (3, 4):
                counts["3/4 ring"] = counts.get("3/4 ring", 0) + 1
            elif len(atom_ring) == 5:
                counts["5 ring"] = counts.get("5 ring", 0) + 1

        # Substituent pairs on aromatic six-membered rings (substituents are attached
        # via non-ring bonds). Counted once, and only if a single pair type exists.
        pair_groups = set()
        for atom_ring in ring_info.AtomRings():
            if len(atom_ring) != 6 or not all(
                mol.GetAtomWithIdx(i).GetIsAromatic() for i in atom_ring
            ):
                continue
            substituted = [
                pos
                for pos, atom_id in enumerate(atom_ring)
                if any(not b.IsInRing() for b in mol.GetAtomWithIdx(atom_id).GetBonds())
            ]
            for pos_i, pos_j in combinations(substituted, 2):
                pair_groups.add(PAIR_GROUPS[min(pos_j - pos_i, 6 - (pos_j - pos_i))])
        if len(pair_groups) == 1:
            counts[pair_groups.pop()] = 1

        return counts


# Decomposes a SMILES string into a single-row DataFrame of Nannoolal group counts
decompose = NannoolalDecomposer(
    "Nannoolal",
    TABLE.index,
    [(group, smarts) for _, group, smarts in sorted(order1_groups)],
)


@nannoolal_gcm.register_property
def dBv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity curve slope for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity curve slope (dBv) of the fuel's components, including the group
        interaction contribution (Nannoolal Eq. (10) with f_GI = 1).
        Converting these sums into dBv requires Nannoolal Eq. (7).
    """
    n = _heavy_atom_counts(fuel)
    dBvi = _contribution_sum(fuel._nannoolal_decomp, "dBv", n, 1.0)
    return Units.Quantity(dBvi, "")


@nannoolal_gcm.register_property
def Tv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity reference temperature for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity reference temperature (Tv) of the fuel's components, including
        the group interaction contribution (Nannoolal Eq. (10) with f_GI = 2).
        Converting these sums into Tv in K requires Nannoolal Eq. (8).
    """
    n = _heavy_atom_counts(fuel)
    Tvi = _contribution_sum(fuel._nannoolal_decomp, "Tv", n, 2.0)
    return Units.Quantity(Tvi, "")

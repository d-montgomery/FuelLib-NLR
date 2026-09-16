"""Units handling and conversions.

This module builds a single, shared :class:`pint.UnitRegistry` (``ureg``) used by
every ``Quantity`` created or accepted throughout FuelLib. Pint only allows
``Quantity`` objects created from the *same* registry to interoperate (arithmetic,
comparisons, conversions), so ``fuellib`` centralizes registry creation here rather
than hiding it behind a re-export shim: callers should access ``fl.units.ureg``
directly (e.g. ``fl.units.ureg.K``) or build quantities via ``fl.units.Quantity``.

The registry is also registered as Pint's *application registry*
(:func:`pint.set_application_registry`), so ``pint.Quantity(...)`` objects created
elsewhere (without importing ``fl.units.ureg``) remain compatible with FuelLib's
quantities.

The registry is case-sensitive (Pint's default). A case-insensitive registry was
considered, but it makes some common abbreviations ambiguous (e.g. ``"kg"``
resolves to ``kilogauss`` instead of ``kilogram``). Instead, the handful of
capitalized temperature unit strings already used across FuelLib's API
(``"Celsius"``, ``"Kelvin"``, ``"Fahrenheit"``) are registered as explicit
aliases below.
"""

import numpy as np
import pint

#: Shared unit registry used by all FuelLib Quantities. Access unit symbols via
#: e.g. ``ureg.K``, ``ureg.Pa``, or build Quantities with :data:`Quantity`.
ureg = pint.UnitRegistry()

# Register capitalized aliases for temperature units used across FuelLib's API.
ureg.define("@alias degree_Celsius = Celsius")
ureg.define("@alias degree_Fahrenheit = Fahrenheit")
ureg.define("@alias kelvin = Kelvin")

# Make this the process-wide default registry so bare ``pint.Quantity(...)``
# objects created elsewhere are still compatible with FuelLib's Quantities.
pint.set_application_registry(ureg)

#: Quantity constructor bound to FuelLib's shared registry, e.g. ``Quantity(1.0, "K")``.
#: Because ``ureg`` is registered as Pint's application registry, ``pint.Quantity``
#: is equivalent to ``ureg.Quantity`` here and is used directly to avoid static
#: type-checker issues with accessing the per-registry ``Quantity`` attribute.
Quantity = pint.Quantity

#: Unit constructor bound to FuelLib's shared registry, e.g. ``Unit("Pa")``.
Unit = pint.Unit


def ustrip(quant: pint.Quantity) -> np.ndarray:
    """
    Strip the unit from a pint Quantity, returning the raw value as a numpy array.

    :param quant: Quantity to strip the unit from.
    :type quant: pint.Quantity
    :return: Raw value without units.
    :rtype: np.ndarray
    """
    return np.array(quant.magnitude)


__all__ = [
    "Quantity",
    "Unit",
    "ureg",
    "ustrip",
]


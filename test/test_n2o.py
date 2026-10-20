#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for N2O solubility, fugacity, and equilibrium concentration functions.

Checks the N2O functions in gasex against Weiss and Price (1980), Mar. Chem.
8, 347-359 (hereafter W&P80), and checks the functions for internal
consistency:

  * N2Osol_SP_pt (K0) reproduces the measured solubilities in W&P80 Table I,
    and its volumetric and gravimetric options agree via seawater density.
  * N2Ofugacity_factor equals the exponential term of W&P80 eq. 9, with
    (B + 2*delta)/RT from eq. 11 (itself checked against eqs. 4 and 6).
  * K0 * (1 - pH2O) * fugfac reproduces the W&P80 moist-air solubility
    function F (Table II) in both volumetric and gravimetric units.
  * eq_SP_pt(gas='N2O') and sol_SP_pt(units='umolkg') are consistent with
    the above and with each other.
  * vpress_sw agrees with W&P80 eq. 10 and with the known vapor pressure of
    pure water (guards against temperature-offset bugs, e.g. 293.15 K
    used in place of 273.15 K).

Run from the repository root with either:
    python test/test_n2o.py
    python -m unittest test.test_n2o -v

@author: Colette Kelly
"""

import os
import sys
import unittest

import numpy as np
import gsw

# test the local copy of gasex, not an installed one (as in test/context.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gasex.sol import N2Osol_SP_pt, O2sol_SP_pt, eq_SP_pt, sol_SP_pt, air_mol_fract
from gasex.fugacity import N2Ofugacity_factor, N2Ofugacity, fugacity_factor
from gasex.phys import vpress_sw


# ---------------------------------------------------------------------------
# Independent reference implementations, written directly from W&P80
# ---------------------------------------------------------------------------

R_WP = 82.05601      # gas constant used by W&P80 [cm3 atm mol-1 K-1]
VBAR_N2O = 32.3      # partial molal volume of N2O, W&P80 p. 349 [cm3 mol-1]

# W&P80 Table II: (A1, A2, A3, A4, B1, B2, B3) for F, eq. 13
F_COEFFS = {
    "volumetric":  (-165.8806, 222.8743, 92.0792, -1.48425,
                    -0.056235, 0.031619, -0.0048472),   # mol L-1 atm-1
    "gravimetric": (-168.2459, 226.0894, 93.2817, -1.48693,
                    -0.060361, 0.033765, -0.0051862),   # mol kg-1 atm-1
}


def kelvin68(t90):
    """ITS-90 deg C -> IPTS-68 absolute temperature [K]."""
    return t90 * 1.00024 + 273.15


def wp_F(SP, t90, units):
    """W&P80 eq. 13: solubility in moist air at 1 atm total pressure."""
    A1, A2, A3, A4, B1, B2, B3 = F_COEFFS[units]
    T = kelvin68(t90) / 100
    return np.exp(A1 + A2 / T + A3 * np.log(T) + A4 * T**2
                  + SP * (B1 + B2 * T + B3 * T**2))


def wp_pH2O(SP, t90):
    """W&P80 eq. 10: vapor pressure of seawater [atm]."""
    T = kelvin68(t90)
    return np.exp(24.4543 - 67.4509 * (100 / T) - 4.8489 * np.log(T / 100)
                  - 0.000544 * SP)


def wp_eq11(T):
    """W&P80 eq. 11: (B + 2*delta)/RT [atm-1], T in K."""
    return -9.4563 / T + 0.04739 - 6.427e-5 * T


def wp_eqs4_6(T):
    """(B + 2*delta)/RT [atm-1] from the virial terms, W&P80 eqs. 4 and 6."""
    B = -905.95 + 4.1685 * T - 0.0052734 * T**2      # cm3 mol-1
    delta = 65.0 - 0.1338 * T                         # cm3 mol-1
    return (B + 2 * delta) / (R_WP * T)


def density_kg_per_L(SP, pt):
    """Seawater density at 0 dbar [kg L-1], computed as gasex.sol does."""
    SA = SP * 35.16504 / 35
    return gsw.rho(SA, gsw.CT_from_pt(SA, pt), 0) / 1000


# grids spanning the W&P80 measurement range (0-40 deg C, 0-40 salinity)
SP_GRID, PT_GRID = np.meshgrid(np.linspace(0, 40, 9), np.linspace(0, 40, 9))
# a few typical oceanic conditions
SP_OCN = np.array([34.5, 35.0, 30.0, 36.5])
PT_OCN = np.array([-1.5, 2.0, 15.0, 28.0])
X_N2O = 335e-9   # dry-air mole fraction used in tests [mol mol-1]


class TestN2OSolubilityCoefficient(unittest.TestCase):
    """N2Osol_SP_pt: K0 of W&P80 eq. 12 / Table II."""

    def test_reproduces_WP80_table_I(self):
        """
        Volumetric K0 vs measured values in W&P80 Table I (10^2 K0,
        mol L-1 atm-1). Table I temperatures are IPTS-68 and are converted to
        ITS-90 here. W&P80 report a fit standard error of 0.14%; allow 0.5%.
        """
        # (S, t68, 10^2 K0)
        table_I = [
            (0.0, 0.29, 5.870), (0.0, 10.00, 4.017), (0.0, 19.97, 2.878),
            (0.0, 30.20, 2.156), (0.0, 40.08, 1.693),
            (18.060, 10.09, 3.629), (18.060, 29.85, 1.985),
            (36.130, 0.30, 4.716), (36.130, 4.93, 3.943),
            (36.130, 10.04, 3.273), (36.130, 20.15, 2.372),
            (36.130, 29.95, 1.812), (36.130, 40.09, 1.425),
        ]
        for S, t68, K0_meas in table_I:
            K0 = 100 * N2Osol_SP_pt(S, t68 / 1.00024)
            with self.subTest(S=S, t68=t68):
                self.assertLess(abs(K0 / K0_meas - 1), 5e-3)

    def test_default_units_are_volumetric(self):
        default = N2Osol_SP_pt(SP_GRID, PT_GRID)
        explicit = N2Osol_SP_pt(SP_GRID, PT_GRID, units="volumetric")
        mixed_case = N2Osol_SP_pt(SP_GRID, PT_GRID, units="Volumetric")
        np.testing.assert_array_equal(default, explicit)
        np.testing.assert_array_equal(default, mixed_case)

    def test_gravimetric_equals_volumetric_over_density(self):
        """
        The two W&P80 K0 fits were made separately, so they agree only to
        within fit error; observed max difference is ~0.013%.
        """
        vol = N2Osol_SP_pt(SP_GRID, PT_GRID, units="volumetric")
        grav = N2Osol_SP_pt(SP_GRID, PT_GRID, units="gravimetric")
        rho = density_kg_per_L(SP_GRID, PT_GRID)
        np.testing.assert_allclose(grav, vol / rho, rtol=2e-4)

    def test_invalid_units_raise(self):
        with self.assertRaises(ValueError):
            N2Osol_SP_pt(35, 10, units="molkg")

    def test_units_is_keyword_only(self):
        with self.assertRaises((TypeError, ValueError)):
            N2Osol_SP_pt(35, 10, "gravimetric")


class TestN2OFugacityFactor(unittest.TestCase):
    """N2Ofugacity_factor: exponential term of W&P80 eq. 9."""

    def test_eq11_matches_virial_coefficients(self):
        """W&P80 eq. 11 is a fit to eqs. 4 and 6; they agree to ~0.015%."""
        T = np.linspace(273.15, 313.15, 41)
        np.testing.assert_allclose(np.exp(wp_eq11(T)), np.exp(wp_eqs4_6(T)),
                                   rtol=5e-4)

    def test_slp_1_is_nonideality_only(self):
        """At P = 1 atm the partial-molal-volume term vanishes."""
        pt = np.linspace(-2, 40, 43)
        np.testing.assert_allclose(N2Ofugacity_factor(pt, slp=1.0),
                                   np.exp(wp_eq11(kelvin68(pt))), rtol=1e-12)

    def test_matches_WP80_eq9_at_other_pressures(self):
        """exp[P(B + 2*delta)/RT + vbar(1 - P)/RT], W&P80 eq. 9."""
        pt = np.linspace(-2, 40, 43)
        T = kelvin68(pt)
        for P in (0.95, 0.98, 1.02, 1.05):
            expected = np.exp(P * wp_eq11(T) + VBAR_N2O * (1 - P) / (R_WP * T))
            with self.subTest(slp=P):
                np.testing.assert_allclose(N2Ofugacity_factor(pt, slp=P),
                                           expected, rtol=1e-12)

    def test_physical_range(self):
        """N2O is slightly non-ideal: factor ~0.995-0.997 at 1 atm."""
        f = N2Ofugacity_factor(np.linspace(-2, 40, 43))
        self.assertTrue(np.all((f > 0.99) & (f < 1.0)))

    def test_dispatch(self):
        pt = np.array([0.0, 10.0, 25.0])
        np.testing.assert_array_equal(fugacity_factor(pt, gas='N2O'),
                                      N2Ofugacity_factor(pt))
        self.assertEqual(fugacity_factor(pt, gas='O2'), 1)


class TestN2OEquilibriumConcentration(unittest.TestCase):
    """Equilibrium N2O built from K0, pH2O, and the fugacity factor."""

    def test_reproduces_WP80_F(self):
        """
        K0 * (1 - pH2O) * fugfac should equal W&P80 F (eq. 9 vs eq. 13) in
        both unit systems. Observed max difference ~0.08% (0-40 deg C,
        S 0-40), within the precision of the W&P80 fits.
        """
        for units in ("volumetric", "gravimetric"):
            for pH2O, label in ((vpress_sw, "vpress_sw"), (wp_pH2O, "W&P eq. 10")):
                built = (N2Osol_SP_pt(SP_GRID, PT_GRID, units=units)
                         * (1 - pH2O(SP_GRID, PT_GRID))
                         * N2Ofugacity_factor(PT_GRID))
                with self.subTest(units=units, pH2O=label):
                    np.testing.assert_allclose(
                        built, wp_F(SP_GRID, PT_GRID, units), rtol=1.5e-3)

    def test_omitting_fugacity_overestimates(self):
        """Without fugfac, concentration is ~0.3-0.5% above F (TRACE issue)."""
        no_fug = (N2Osol_SP_pt(SP_OCN, PT_OCN, units="gravimetric")
                  * (1 - wp_pH2O(SP_OCN, PT_OCN)))
        ratio = no_fug / wp_F(SP_OCN, PT_OCN, "gravimetric")
        self.assertTrue(np.all((ratio > 1.002) & (ratio < 1.006)))

    def test_eq_SP_pt_uses_gravimetric_K0(self):
        """eq_SP_pt(umolkg) = 1e6 * K0_grav * x' * (P - pH2O), no density."""
        pH2O = vpress_sw(SP_OCN, PT_OCN)
        K0g = N2Osol_SP_pt(SP_OCN, PT_OCN, units="gravimetric")
        for P in (1.0, 0.97):
            expected = 1e6 * K0g * X_N2O * (P - pH2O)
            result = eq_SP_pt(SP_OCN, PT_OCN, gas='N2O', slp=P,
                              chi_atm=X_N2O, units='umolkg')
            with self.subTest(slp=P):
                np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_eq_SP_pt_with_fugacity_matches_WP80_F(self):
        result = (eq_SP_pt(SP_GRID, PT_GRID, gas='N2O', slp=1.0,
                           chi_atm=X_N2O, units='umolkg')
                  * fugacity_factor(PT_GRID, gas='N2O'))
        expected = 1e6 * wp_F(SP_GRID, PT_GRID, "gravimetric") * X_N2O
        np.testing.assert_allclose(result, expected, rtol=1.5e-3)

    def test_eq_SP_pt_unit_conversions(self):
        umolkg = eq_SP_pt(SP_OCN, PT_OCN, gas='N2O', chi_atm=X_N2O, units='umolkg')
        molar = eq_SP_pt(SP_OCN, PT_OCN, gas='N2O', chi_atm=X_N2O, units='M')
        rho = density_kg_per_L(SP_OCN, PT_OCN)
        np.testing.assert_allclose(molar, umolkg * rho * 1e-6, rtol=1e-10)

    def test_N2Ofugacity_times_K0(self):
        """K0 * N2Ofugacity (atm) equals eq_SP_pt * fugacity_factor."""
        f_atm = N2Ofugacity(SP_OCN, PT_OCN, slp=1.0, xn2o=X_N2O, units="atm")
        C_from_f = 1e6 * N2Osol_SP_pt(SP_OCN, PT_OCN, units="gravimetric") * f_atm
        C_eq = (eq_SP_pt(SP_OCN, PT_OCN, gas='N2O', slp=1.0, chi_atm=X_N2O,
                         units='umolkg') * fugacity_factor(PT_OCN, gas='N2O'))
        np.testing.assert_allclose(C_from_f, C_eq, rtol=1e-12)

    def test_typical_values(self):
        """Sanity range: ~14 nmol/kg at 2 C, ~6.5 nmol/kg at 28 C (335 ppb)."""
        C = 1e3 * eq_SP_pt(np.array([34.5, 34.5]), np.array([2.0, 28.0]),
                           gas='N2O', chi_atm=X_N2O, units='umolkg')
        self.assertTrue(13.5 < C[0] < 15.0)
        self.assertTrue(5.5 < C[1] < 6.5)


class TestSolSPptUnits(unittest.TestCase):
    """sol_SP_pt(units='umolkg') must be K0 converted to umol kg-1 atm-1."""

    def test_umolkg_equals_molar_over_density(self):
        rho = density_kg_per_L(SP_OCN, PT_OCN)
        for gas, rtol in (('CO2', 1e-12), ('CH4', 1e-12), ('O2', 1e-12),
                          ('Ar', 1e-12), ('N2O', 2e-4)):
            molar = sol_SP_pt(SP_OCN, PT_OCN, gas=gas, units="M")
            umolkg = sol_SP_pt(SP_OCN, PT_OCN, gas=gas, units="umolkg")
            with self.subTest(gas=gas):
                np.testing.assert_allclose(umolkg, 1e6 * molar / rho, rtol=rtol)

    def test_N2O_umolkg_is_gravimetric_K0(self):
        np.testing.assert_allclose(
            sol_SP_pt(SP_OCN, PT_OCN, gas='N2O', units="umolkg"),
            1e6 * N2Osol_SP_pt(SP_OCN, PT_OCN, units="gravimetric"), rtol=1e-12)

    def test_N2O_umolkg_magnitude(self):
        """~4.3e4 umol kg-1 atm-1 at 2 C (guards the old 1e12 unit error)."""
        K0 = sol_SP_pt(34.5, 2.0, gas='N2O', units="umolkg")
        self.assertTrue(4.0e4 < K0 < 4.6e4)

    def test_O2_rebuilds_garcia_gordon(self):
        """K0(umolkg) * x_O2 * (1 - pH2O) recovers O2sol_SP_pt."""
        rebuilt = (sol_SP_pt(SP_OCN, PT_OCN, gas='O2', units="umolkg")
                   * air_mol_fract(gas='O2') * (1 - vpress_sw(SP_OCN, PT_OCN)))
        np.testing.assert_allclose(rebuilt, O2sol_SP_pt(SP_OCN, PT_OCN), rtol=1e-5)


class TestVaporPressure(unittest.TestCase):

    def test_vpress_sw_matches_WP80_eq10(self):
        """Within 0.2% of pH2O over 0-40 C, S 0-40 (observed ~0.16%)."""
        np.testing.assert_allclose(vpress_sw(SP_GRID, PT_GRID),
                                   wp_pH2O(SP_GRID, PT_GRID), rtol=2e-3)

    def test_pure_water_reference_values(self):
        """
        Saturation vapor pressure of pure water: 6.112 hPa at 0 C,
        23.39 hPa at 20 C (e.g. Wagner & Pruss 2002). A 20 K temperature
        offset would roughly triple these values.
        """
        hPa_per_atm = 1013.25
        for t, p_ref in ((0.0, 6.112), (20.0, 23.39)):
            with self.subTest(t=t):
                self.assertLess(abs(vpress_sw(0, t) * hPa_per_atm / p_ref - 1), 2e-3)
                self.assertLess(abs(wp_pH2O(0, t) * hPa_per_atm / p_ref - 1), 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

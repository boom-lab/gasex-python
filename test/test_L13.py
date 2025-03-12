#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 11 2025

To run unit tests: from the gasex-python directory, run:
```bash
python3 -m unittest discover test
```

@author: Colette Kelly
"""

import unittest
from gasex import airsea
from gasex.sol import sol_SP_pt

class TestL13Function(unittest.TestCase):
    """
    Unit tests for the L13 function in the gasex module.
    """

    def test_L13_Ar(self):
        """
        Test L13 function with gas='Ar'. 
        This test checks the expected values of Fd, Fc, Fp, Deq, and Ks 
        when the gas is Argon (Ar), with given input parameters.
        """
        tolx = 1e-10
        (Fd, Fc, Fp, Deq, Ks) = airsea.L13(0.01410, 5, 35, 10, slp=1.0, gas='Ar', rh=0.9)

        # Expected values
        expected_Fd = -5.26320104779024e-09
        expected_Fc = 1.3605129856657824e-10
        expected_Fp = -6.007937936550423e-10
        expected_Deq = 0.0014239320927325997
        expected_Ks = 2.0374270624087916e-05

        # Assertions with tolerance
        self.assertTrue(abs(Fd / expected_Fd - 1) < tolx)
        self.assertTrue(abs(Fc / expected_Fc - 1) < tolx)
        self.assertTrue(abs(Fp / expected_Fp - 1) < tolx)
        self.assertTrue(abs(Deq / expected_Deq - 1) < tolx)
        self.assertTrue(abs(Ks / expected_Ks - 1) < tolx)

    def test_L13_N2O(self):
        """
        Test L13 function with gas='N2O'.
        Check that the outputs with and without pressure_mode are equivalent.
        """
        tolx = 1e-10
        s = sol_SP_pt(35,10,chi_atm=3.37e-7, gas="N2O",units="mM")
        C = 13e-6
        pN2Osw = C/s

        (Fd, Fc, Fp, Deq, Ks) = airsea.L13(C,5,35,10,slp=1.0,gas='N2O',rh=1.0, chi_atm=3.37e-7)
        (Fd, Fc, Fp) = (Fd*1e6*86400, Fc*1e6*86400, Fp*1e6*86400)

        (Fd_p, Fc_p, Fp_p, Deq_p, Ks_p) = airsea.L13(pN2Osw,5,35,10,slp=1.0,gas='N2O',rh=1.0, chi_atm=3.37e-7,pressure_mode = True)
        (Fd_p, Fc_p, Fp_p) = (Fd_p*1e6*86400, Fc_p*1e6*86400, Fp_p*1e6*86400)

        # Expected values
        expected_Fd = -2.888498915956542
        expected_Fc = 0.00042449404737498347
        expected_Fp = -0.4571051450856986
        expected_Deq = 0.0009693558952097625
        expected_Ks = 1.636503328219589e-05

        # Assertions with tolerance
        self.assertTrue(abs(Fd / expected_Fd - 1) < tolx)
        self.assertTrue(abs(Fc / expected_Fc - 1) < tolx)
        self.assertTrue(abs(Fp / expected_Fp - 1) < tolx)
        self.assertTrue(abs(Deq / expected_Deq - 1) < tolx)
        self.assertTrue(abs(Ks / expected_Ks - 1) < tolx)

        # Assertions with tolerance
        self.assertTrue(abs(Fd_p / expected_Fd - 1) < tolx)
        self.assertTrue(abs(Fc_p / expected_Fc - 1) < tolx)
        self.assertTrue(abs(Fp_p / expected_Fp - 1) < tolx)
        self.assertTrue(abs(Deq_p / expected_Deq - 1) < tolx)
        self.assertTrue(abs(Ks_p / expected_Ks - 1) < tolx)

if __name__ == "__main__":
    unittest.main()

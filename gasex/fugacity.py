#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 2024

Functions for calculating gas fugacity and fugacity correction factors.

@author: Colette Kelly
"""

import numpy as np
from ._utilities import match_args_return
from gasex.phys import K0 as K0
from gasex.phys import vpress_sw

@match_args_return
def fugacity_factor(pt,gas=None,slp=1.0):
    g_up = gas.upper()

    if g_up == 'CO2':
        #fugfac = CO2fugacity_factor(pt,slp=slp)
        return 1
    elif g_up == 'N2O':
        return N2Ofugacity_factor(pt,slp=slp)
    else:
        return 1

@match_args_return
def N2Ofugacity_factor(pt,slp=1.0):
    """
    DESCRIPTION
    -------------
    Calculate the combined fugacity correction factor for N2O.

    Returns the exponential term of Weiss and Price (1980), eq. (9):

        F = K0 * (P - pH2O) * exp[ P*(B + 2*delta)/(R*T) + v*(1 - P)/(R*T) ]

    i.e. fugfac = exp[ P*(B + 2*delta)/(R*T) + v*(1 - P)/(R*T) ], which
    combines the gas non-ideality of N2O in air (first term; (B + 2*delta)/RT
    from eq. (11)) with the pressure dependence of K0 via the partial molal
    volume v (second term; eq. (1)). The second term is zero at P = 1 atm.

    K0 * x' * (P - pH2O) * fugfac is therefore the equilibrium concentration
    at total pressure P, where K0 is from gasex.sol.N2Osol_SP_pt and x' is
    the dry-air mole fraction of N2O.

    INPUTS:
    ----------
    pt      Potential temperature (ITS-90) referenced to 1 atm   [deg C]
    slp     Sea level pressure                                   [atm] (default: 1.0)

    OUTPUT:
    ----------
    fugfac  Combined fugacity correction factor for N2O, Weiss and Price
            (1980) eq. (9) (dimensionless)

    AUTHOR: Colette Kelly
    """
    # Weiss and Price calculation uses R = 0.08205601 L*atm/mol/K
    # define local R here instead of using global R from gasex.phys
    R = 0.08205601

    # Weiss and Price: partial molal volume of N2O is 32.3 cm3/mol
    v=32.3
    # need to convert from cm3 to L for exponential term 2 to be dimensionless
    v = v/1000

    # Convert temperature to IPTS-68 to match temperatures used in Weiss and Price
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # Calculate exponential term 1: P*(B + 2*delta)/RT (Weiss and Price 1980, eqns. (9) and (11))
    eterm1 = slp*(-9.4563/y + 0.04739 - 6.427*10**-5*y)
    
    # Calculate exponential term 2: v*(1 - P)/RT (Weiss and Price 1980, eqns. (1) and (9))
    eterm2 = (1-slp)*v/(R*y)
    
    # combined fugacity correction factor, Weiss and Price 1980, eqn. (9)
    fugfac = np.exp(eterm1+eterm2)

    return fugfac

@match_args_return
def N2Ofugacity(SP,pt,slp=1.0,xn2o=338e-9,rh=1.0,units = "natm"):
    """
    DESCRIPTION
    -------------
    Calculate the fugacity of N2O in seawater.

    Fugacity accounts for non-ideal gas behavior and is used in air-sea gas exchange calculations.
    This function implements the Weiss and Price (1980) method to compute N2O fugacity, incorporating 
    corrections for seawater vapor pressure and gas non-ideality.

    INPUTS:
    ----------
    SP      Practical Salinity (PSS-78) (dimensionless)
    pt      Potential temperature (ITS-90) referenced to 1 atm   [deg C]
    slp     Sea level pressure                                   [atm] (default: 1.0)
    xn2o    Mixing ratio of N2O in dry air                      [mol/mol] (default: 338e-9)
    rh      Relative humidity                                    [fraction] (default: 1.0)
    units   Units of fugacity ("natm" for nanonatmospheres, "atm" for atmospheres) (default: "natm")

    OUTPUT:
    ----------
    f       Fugacity of N2O in the specified units

    AUTHOR: Colette Kelly
    """
    # calculate fugacity factor
    fugfac = N2Ofugacity_factor(pt,slp=slp)

    # convert xn2o from mol/mol to ppb
    xn2o = xn2o*1e9
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)*rh # vapor pressure of water
    
    # fugacity, Weiss and Price 1980: eqn. (7) at slp = 1 atm; for slp != 1 atm,
    # fugfac also carries the v*(1 - P)/RT term, so f = F/K0 from eqn. (9)
    # (i.e. K0 * f is the equilibrium concentration)
    f = xn2o*(slp-vp_sw)*fugfac
    
    # Convert to atmospheres if specified
    if units.lower() == "atm":
        f = f*10**-9

    return f

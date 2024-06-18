#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 2024

Functions for calculating gas atmospheric partial pressure,
using fugacity when available.

@author: C. Kelly
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
    Calculate the fugacity of N2O in seawater.

    Parameters:
    - SP: Practical Salinity  (PSS-78) (unitless)
    - pt: potential temperature (ITS-90) referenced
             to one standard atmosphere (0 dbar).
    - slp: Sea level pressure (atm) (default: 1.0)
    - xn2o: Mixing ratio of N2O  in dry air (ppb) (default: 333 ppb)
    - units: Units of fugacity ("natm" for nanonatmospheres, "atm" for atmospheres) (default: "natm")

    Returns:
    - f: Fugacity of N2O in the specified units
    """
    # Weiss and Price calculation uses R = 0.08205601 L*atm/mol/K
    # define local R here instead of using global R from gasex.phys
    R = 0.08205601

    # Weiss and Price: partial molal volume of N2O is 32.2 cm3/mol
    v=32.3
    # need to convert from cm3 to L for exponential term 2 to be dimensionless
    v = v/1000

    # Convert temperature to IPTS-68 to match temperatures used in Weiss and Price
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # Calculate exponential term 1 (Weiss and Price )1980, eqn. (11))
    eterm1 = slp*(-9.4563/y + 0.04739 - 6.427*10**-5*y)
    
    # Calculate exponential term 2 (Weiss and Price 1980, eqn. (1))
    eterm2 = (1-slp)*v/(R*y)
    
    # fugacity, Weiss and Price 1980, eqn. (7)
    fugfac = np.exp(eterm1+eterm2)

    return fugfac

@match_args_return
def N2Ofugacity(SP,pt,slp=1.0,xn2o=338e-9,v=32.3,rh=1.0,units = "natm"):
    """
    Calculate the fugacity of N2O in seawater.

    Parameters:
    - SP: Practical Salinity  (PSS-78) (unitless)
    - pt: potential temperature (ITS-90) referenced
             to one standard atmosphere (0 dbar).
    - slp: Sea level pressure (atm) (default: 1.0)
    - xn2o: Mixing ratio of N2O  in dry air (ppb) (default: 333 ppb)
    - units: Units of fugacity ("natm" for nanonatmospheres, "atm" for atmospheres) (default: "natm")

    Returns:
    - f: Fugacity of N2O in the specified units
    """
    # calculate fugacity factor
    fugfac = N2Ofugacity_factor(pt,slp=slp)

    # convert xn2o from mol/mol to ppb
    xn2o = xn2o*1e9
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)*rh # vapor pressure of water
    
    # fugacity, Weiss and Price 1980, eqn. (7)
    f = xn2o*(slp-vp_sw)*fugfac
    
    # Convert to atmospheres if specified
    if units.lower() == "atm":
        f = f*10**-9

    return f

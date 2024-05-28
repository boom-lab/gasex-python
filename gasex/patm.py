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
from gasex.phys import vpress_sw, R
from gasex.sol import air_mol_fract


@match_args_return
def patm(SP,pt,gas=None,slp=1.0,rh=1.0,chi_atm=None,units="atm"):
    g_up = gas.upper()
    vp_sw = vpress_sw(SP,pt)*rh
    if chi_atm is None:
        chi_atm = air_mol_fract(gas=gas)
    
    gas_list = ['O2','HE','NE','AR','KR','XE','N2','CO2','CH4','CO','H2']

    if g_up in gas_list:
        p = chi_atm*(slp - vp_sw)
    elif gas == 'N2O':
        p = N2Ofugacity(SP,pt,slp=slp,xn2o=chi_atm,v=32.3,rh=rh,units = "atm")
    else:
        raise ValueError(f"{gas} is not supported. Must be in {gas_list}")

    if units not in ("atm", "natm", "pa"):
        raise ValueError("units: units must be \'M\','uM', 'nM', or \'umolkg\'")

    if units == "atm":
        p = p
    elif units =="natm":
        p =  p*1e9
    elif units =="pa":
        p =  p * 101325

    return p

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
    # convert xn2o from mol/mol to ppb
    xn2o = xn2o*1e9
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)*rh # vapor pressure of water
    
    # Convert temperature to IPTS-68
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # Calculate exponential term 1 (Weiss and Price )1980, eqn. (11))
    eterm1 = slp*(-9.4563/y + 0.04739 - 6.427*10**-5*y)
    
    # Calculate exponential term 2 (Weiss and Price 1980, eqn. (1))
    eterm2 = (1-slp)*v/(R*y)
    
    # fugacity, Weiss and Price 1980, eqn. (7)
    f = xn2o*(slp-vp_sw)*np.exp(eterm1+eterm2)
    
    # Convert to atmospheres if specified
    if units.lower() == "atm":
        f = f*10**-9

    return f

if __name__=="__main__":
	f = patm(33,0,gas="N2O",slp=1,rh=1,chi_atm=338e-9,units="natm")
	print(f)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024

Functions for calculating N2O equilibrium concentration,
partial pressure at the moist interface, and mixing ratio in water,
taking into account water vapor pressure and non-ideal effects.

@author: C. Kelly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gsw import pt_from_CT,SP_from_SA,CT_from_pt,rho
from gasex._utilities import match_args_return
from gasex.phys import K0 as K0
from gasex.phys import vpress_sw, R
from gasex.sol import N2Osol_SP_pt

@match_args_return
def N2Ofugacity(SP,pt,slp=1.0,xn2o=338,units = "natm"):
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
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt) # vapor pressure of water
    
    # Convert temperature to IPTS-68
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # exponential term, Weiss and Price 1980, eqn. (11)
    eterm = -9.4563/y + 0.04739 - 6.427*10**-5*y
    
    # fugacity, Weiss and Price 1980, eqn. (7)
    f = xn2o*(slp-vp_sw)*np.exp(slp*eterm)
    
    # Convert to atmospheres if specified
    if units.lower() == "atm":
        f = f*10**-9

    return f

def N2OCeq(SP, pt, slp=1.0, xn2o=338, v=32.3, watervapor=True, nonideal=True):
    """
    Calculate the equilibrium concentration of N2O in seawater,
    taking into account the vapor pressure of water and non-ideal effects.

    Parameters:
    - SP: Practical Salinity (PSS-78) (unitless)
    - pt: Potential temperature (ITS-90) referenced to one standard atmosphere (0 dbar) (°C)
    - slp: Sea level pressure (atm) (default: 1.0)
    - xn2o: Mixing ratio of N2O  in dry air (ppb) (default: 333 ppb)
    - v: Partial molal volume of N2O (cm3/mol) (default: 32.3 cm3/mol, Weiss 1974)
    - watervapor: Include effect of water vapor (default: True)
    - nonideal: Include non-ideal behavior of N2O (default: True)

    Returns:
    - Ceq: Equilibrium concentration of N2O (nmol/L), including vapor pressure of water and non-ideal effects
    """  
   
    # Convert partial molal volume from cm3/mol to L/mol
    v = v/1000.0

    # Ideal gas constant (L*atm/mol/K)
    R = 0.08205691 # ideal gas constant (L*atm/mol/K)
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)

    # Convert temperature to IPTS-68
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # Calculate equilibrium constant, mol/L/atm
    k0 = N2Osol_SP_pt(SP,pt)
    
    # Calculate exponential term 1 (Weiss and Price 1980, eqn. (11))
    eterm1 = -9.4563/y + 0.04739 - 6.427*10**-5*y
    
    # Calculate exponential term 2 (Weiss and Price 1980, eqn. (1))
    eterm2 = (1-slp)*v/(R*y)

    # Calculate F (Weiss and Price 1980, eqn. (9))
    # F is a simplification of C*=K0*f*exp[(1-P)v/RT] (Weiss and Price 1980, eqn. (1))
    F = k0*(slp-vp_sw)*np.exp(eterm1+eterm2)

    if (watervapor==True)&(nonideal==True):
        # Weiss and Price 1980, eqn. (8)
        Ceq = xn2o*F
    elif (watervapor==True)&(nonideal==False):
        # exclude non-ideal effects
        Ceq = k0*xn2o*(slp - vp_sw)
    elif (watervapor==False)&(nonideal==False):
        # exclude vapor pressure of water and non-ideal effects
        Ceq = k0*xn2o*slp
    elif (watervapor==False)&(nonideal==True):
        Ceq = xn2o*k0*slp*np.exp(eterm1+eterm2)
    
    return Ceq

def pN2Oatm(SP,pt,slp=1.0,xn2o=338, units = "natm", v=32.3, watervapor=True, nonideal=True):
    """
    Calculate the partial pressure of N2O at the moist interface.

    Parameters:
    - SP: Practical Salinity  (PSS-78) (unitless)
    - pt: potential temperature (ITS-90) referenced
             to one standard atmosphere (0 dbar).
    - slp: Sea level pressure (atm) (default: 1.0)
    - xn2o: xn2o: Mixing ratio of N2O  in dry air (ppb) (default: 333 ppb)
    - units: Units of fugacity ("natm" for nanonatmospheres, "atm" for atmospheres) (default: "natm")
    - v: Partial molal volume of N2O (cm3/mol) (default: 32.3 cm3/mol, Weiss 1974)
    - watervapor: Include effect of water vapor (default: True)
    - nonideal: Include non-ideal behavior of N2O (default: True)

    Returns:
    - pN2Oatm1: Partial pressure of N2O at the moist interface (natm or atm), including vapor pressure of water and non-ideal effects
    - pN2Oatm2: Partial pressure of N2O at the moist interface (natm or atm), including vapor pressure of water and but excluding non-ideal effects
    - pN2Oatm3: Partial pressure of N2O at the moist interface (natm or atm), excluding vapor pressure of water and non-ideal effects
    """
    
    # Convert partial molal volume from cm3/mol to L/mol
    v = v/1000.0
    
    # Ideal gas constant (L*atm/mol/K)
    R = 0.08205691
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)
    
    # Convert temperature to IPTS-68
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py
    
    # Calculate exponential term 1 (Weiss and Price 1980, eqn. (11))
    eterm1 = -9.4563/y + 0.04739 - 6.427*10**-5*y

    # Calculate exponential term 2 (Weiss and Price 1980, eqn. (1))
    eterm2 = (1-slp)*v/(R*y)

    if (watervapor==True)&(nonideal==True):
        # pN2Oatm = Ceq/k0 = x'(P-pH2O)exp(P(B+2d)/RT+(1-P)v/RT)
        pN2Oatm = xn2o*(slp - vp_sw)*np.exp(eterm1+eterm2)
    elif (watervapor==True)&(nonideal==False):
        # pN2Oatm = x'(P-pH2O)
        pN2Oatm = xn2o*(slp - vp_sw)
    elif (watervapor==False)&(nonideal==False):
        # pN2Oatm = x'P
        pN2Oatm = xn2o*slp
    elif (watervapor==False)&(nonideal==True):
        # pN2Oatm = Ceq/k0 = x'(P-pH2O)exp(P(B+2d)/RT+(1-P)v/RT)
        pN2Oatm = xn2o*slp*np.exp(eterm1+eterm2)

    # Convert to atmospheres if specified
    if units.lower() == "atm":
        pN2Oatm = pN2Oatm*10**-9
    
    return pN2Oatm

def pN2Osw(SP,pt,C, units = "natm"):
    """
    Calculate the fugacity of N2O in seawater.

    Parameters:
    - SP: Practical Salinity  (PSS-78) (unitless)
    - pt: potential temperature (ITS-90) referenced
             to one standard atmosphere (0 dbar).
    - C: concentration (nmol/L)

    Returns:
    - pN2Osw: partial pressure of dissolved N2O (natm or atm)
    """    
    
    # Calculate equilibrium constant, mol/L/atm
    k0 = N2Osol_SP_pt(SP,pt)
    
    # calculate partial pressure of N2O in seawater (natm)
    pN2Osw = C/k0
    
    # Convert to atmospheres if specified
    if units.lower() == "atm":
        pN2Osw = pN2Osw*10**-9

    return pN2Osw

def xN2Osw(SP, pt, C, slp=1.0, v=32.3, watervapor=True, nonideal=True):
    """
    Calculate the mixing ratio of N2O corresponding to a given concentration in seawater.

    Parameters:
    - SP: Practical Salinity (PSS-78) (unitless)
    - pt: Potential temperature (ITS-90) referenced to one standard atmosphere (0 dbar) (°C)
    - C: Concentration of N2O (nmol/L)
    - slp: Sea level pressure (atm) (default: 1.0)
    - v: Partial molal volume of N2O (cm3/mol) (default: 32.3 cm3/mol, Weiss 1974)
    - watervapor: Include effect of water vapor (default: True)
    - nonideal: Include non-ideal behavior of N2O (default: True)

    Returns:
    - k0: Equilibrium constant (mol/L/atm)
    - F: Simplification of C*=K0*f*exp[(1-P)v/RT] (Weiss and Price 1980, eqn. (1)) (mol/L)
    - xN2O1: Mixing ratio of N2O (nmol/mol or ppb), including vapor pressure of water and non-ideal effects
    - xN2O2: Mixing ratio of N2O (nmol/mol or ppb), including vapor pressure of water and but excluding non-ideal effects
    - xN2O3: Mixing ratio of N2O (nmol/mol or ppb), excluding vapor pressure of water and non-ideal effects
    """  
   
    # Convert partial molal volume from cm3/mol to L/mol
    v = v/1000.0

    # Ideal gas constant (L*atm/mol/K)
    R = 0.08205691 # ideal gas constant (L*atm/mol/K)
    
    # Calculate the vapor pressure of water
    vp_sw = vpress_sw(SP,pt)

    # Convert temperature to IPTS-68
    pt68 = pt * 1.00024
    y = pt68 + K0 # K0 is 0 Celsius in Kelvin, imported from phys.py

    # Calculate equilibrium constant, mol/L/atm
    k0 = N2Osol_SP_pt(SP,pt)
    
    # Calculate exponential term 1 (Weiss and Price 1980, eqn. (11))
    eterm1 = -9.4563/y + 0.04739 - 6.427*10**-5*y
    
    # Calculate exponential term 2 (Weiss and Price 1980, eqn. (1))
    eterm2 = (1-slp)*v/(R*y)
    # Calculate F (Weiss and Price 1980, eqn. (9))
    # F is a simplification of C*=K0*f*exp[(1-P)v/RT] (Weiss and Price 1980, eqn. (1))    
    F = k0*(slp-vp_sw)*np.exp(eterm1+eterm2)

    if (watervapor==True)&(nonideal==True):
        # Weiss and Price 1980, eqn. (8)
        xN2O = C/F
    elif (watervapor==True)&(nonideal==False):
        # exclude non-ideal effects
        xN2O = C/(k0*(slp - vp_sw))
    elif (watervapor==False)&(nonideal==False):
        # exclude vapor pressure of water and non-ideal effects
        xN2O = C/(k0*slp)
    elif (watervapor==False)&(nonideal==True):
        F = k0*slp*np.exp(eterm1+eterm2)
        xN2O = C/F
  
    return k0, F, xN2O

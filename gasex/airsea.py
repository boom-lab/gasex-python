#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:55:36 2017

@author: dnicholson
"""

from gasex.phys import vpress_sw
from gasex.diff import schmidt
from gasex.sol import sol_SP_pt,eq_SP_pt

def kgas(u10,Sc,param="W14"):
    """
    DESCRIPTION:
    ----------
    Calculate gas transfer piston velocity fore diffusive gas exchange 
    as a function of schmidt number using a range of published wind speed based
    parameterizations     
    
    INPUTS:
    ----------
    u10       10-m wind speed (m/s)
    Sc        Schmidt number 
    param     abbreviation for parameterization:
           W14  = Wanninkhof 2014
           W92a = Wanninkhof 1992 - averaged winds
           W92b = Wanninkhof 1992 - instantaneous or steady winds
           Sw07 = Sweeney et al. 2007
           Ho06 = Ho et al. 2006
           Ng00 = Nightingale et al. 2000
           LM86 = Liss and Merlivat 1986
    OUTPUT:
    ---------- 
    gas transfer velocity, k, in m s-1
    
    References
    ----------
      Wanninkhof, R. (2014). Relationship between wind speed and gas exchange
      over the ocean revisited. Limnol. Oceanogr. Methods, 12(6), 351-362.
     
      Wanninkhof, R. (1992). Relationship between wind speed and gas exchange
      over the ocean. J. Geophys. Res, 97(25), 7373-7382.
     
      Sweeney, C., Gloor, E., Jacobson, A. R., Key, R. M., McKinley, G.,
      Sarmiento, J. L., & Wanninkhof, R. (2007). Constraining global air-sea
      gas exchange for CO2 with recent bomb 14C measurements. Global
      Biogeochem. Cy.,21(2).
     
      Ho, D. T., Law, C. S., Smith, M. J., Schlosser, P., Harvey, M., & Hill,
      P. (2006). Measurements of air?sea gas exchange at high wind speeds in
      the Southern Ocean: Implications for global parameterizations. Geophys.
      Res. Lett., 33(16).
     
      Nightingale, P. D., Malin, G., Law, C. S., Watson, A. J., Liss, P. S.,
      Liddicoat, M. I., et al. (2000). In situ evaluation of air-sea gas
      exchange parameterizations using novel conservative and volatile tracers.
      Global Biogeochem. Cy., 14(1), 373-387.
    """
    
    p_upper = param.upper()
    # coefficient for quadratic params
    if p_upper in ("W14","W92","SW07","HO06","W92_AVE"):
        A_dict = {"W14": 0.251, \
              "W92": 0.31, \
              "SW07": 0.27, \
              "HO06": 0.254, \
              "W92_ave": 0.39}
        k_cm = A_dict[p_upper] * u10**2 * (Sc / 660)**(-0.5) # cm/hr       
    elif p_upper == "NG00":
        k600 = 0.222 * u10**2 + 0.333 * u10
        k_cm = k600 * (Sc / 600)^(-0.5)
    else:
        raise ValueError(param + " is not a supporte parameterization" )
    k = k_cm / (100*60*60)
    return k



def fas(C,u10,SP,T,slp=1.0,gas=None,param="W14",rh=1):
    """
    DESCRIPTION
    -------------
    Diffusive gas flux across the air-sea interface in mmol m-2 s-1
    
    F = k (C - Ceq_slp)
    
    INPUTS:
    ----------
    C         Surface dissolved gas concentration       [mmol m-3 == mol L-1]        
    u10       10-m wind speed                           [m s-1]
    SP        Practical Salinity                         --
    T         Surface water temperature                 [deg C ]
    slp       Sea-level pressure                        [atm]
    gas       abbr. for gas of interest
    param     abbr. for the air-sea parameterization    
    rh        fractional relative humidity (1 = 100%)
    
    OUTPUT:
    ---------- 
    air sea gas flux (positive out of ocean) in mmol m-2 s-1
    
    AUTHOR: David Nicholson
    """
    slp_corr = (slp - vpress_sw(SP,T)) / (1 - vpress_sw(SP,T))
    C_eq = eq_SP_pt(SP,T,gas=gas)
    Sc = schmidt(SP,T,gas=gas)
    k = kgas(u10,Sc,param)
    Fd = k * (C - C_eq * slp_corr)  
    return Fd

def fas_pC(pC_w,pC_a,u10,SP,T,gas=None,param="W14"): 
    """
    DESCRIPTION
    -------------
    Diffusive gas flux across the air-sea interface in mmol m-2 s-1
    
    INPUTS:
    ----------
    pC_w      dissolved partial pressure (or fugacity)  [uatm]    
    pC_a      air-side partial pressure (or fugacity)   [uatm]
    u10       10-m wind speed                           [m s-1]
    SP        Practical Salinity                         --
    T         Surface water temperature                 [deg C ]
    slp       Sea-level pressure                        [atm]
    gas       abbr. for gas of interest
    param     abbr. for the air-sea parameterization    
    
    OUTPUT:
    ---------- 
    air sea gas flux (positive out of ocean) in mmol m-2 s-1
    
    AUTHOR: David Nicholson
    """
    
    K0 = sol_SP_pt(SP,T,gas=gas)
    Sc = schmidt(SP,T,gas=gas)
    k = kgas(u10,Sc,param)
    Fd = k * K0 * (pC_w - pC_a) / 1e6
    return Fd
    
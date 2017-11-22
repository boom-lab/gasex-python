#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:55:36 2017

@author: dnicholson
"""
import numpy as np
from gasex.phys import vpress_sw, R, cdlp81, atm2pa, K0
from gasex.diff import schmidt,diff
from gasex.sol import sol_SP_pt,eq_SP_pt, air_mol_fract
from gsw import rho, CT_from_pt
from ._utilities import match_args_return

def kgas(u10,Sc,*,param="W14"):
    """
    DESCRIPTION:
    ----------
    Calculate gas transfer piston velocity for diffusive gas exchange 
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



def fsa(C,u10,SP,T,*,slp=1.0,gas=None,param="W14",rh=1):
    """
    DESCRIPTION
    -------------
    Diffusive gas flux across the air-sea interface in mol m-2 s-1
    
    F = k (C - Ceq_slp)
    
    INPUTS:
    ----------
    C         Surface dissolved gas concentration       [mol m-3 == mmol L-1]        
    u10       10-m wind speed                           [m s-1]
    SP        Practical Salinity                         --
    T         Surface water temperature                 [deg C ]
    slp       Sea-level pressure                        [atm]
    gas       abbr. for gas of interest
    param     abbr. for the air-sea parameterization    
    rh        fractional relative humidity (1 = 100)
    
    OUTPUT:
    ---------- 
    air sea gas flux (positive out of ocean) in mol m-2 s-1
    
    AUTHOR: David Nicholson
    """
    slp_corr = (slp - vpress_sw(SP,T)) / (1 - vpress_sw(SP,T))
    # equilibrium conc. [mol L-1 == mmol m-3]
    C_eq = eq_SP_pt(SP,T,gas=gas)
    Sc = schmidt(SP,T,gas=gas)
    # piston velocity [m s-1]
    k = kgas(u10,Sc,param)
    Fd = k * (C - C_eq * slp_corr)  
    return Fd

def fsa_pC(pC_w,pC_a,u10,SP,T,*,gas=None,param="W14"): 
    """
    DESCRIPTION
    -------------
    Diffusive gas flux across the air-sea interface in mol m-2 s-1
    
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
    
    # K0 in mmol L-1 atm-1 == mol m-3 atm-1
    K0 = sol_SP_pt(SP,T,gas=gas,units="mM")
    Sc = schmidt(SP,T,gas=gas)
    # piston velocity m s-1
    k = kgas(u10,Sc,param)
    # m s-1 * mol m-3 atm -1 * atm == mol m-2 s-1
    Fd = k * K0 * (pC_w - pC_a) / 1e6
    return Fd

@match_args_return
def L13(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None):
    """
    % fas_L13: Function to calculate air-sea fluxes with Liang 2013
    % parameterization
    %
    % USAGE:-------------------------------------------------------------------
    % [Fd, Fc, Fp, Deq, k] = fas_L13(C,u10,S,T,slp,gas,rh)
    % [Fd, Fc, Fp, Deq, k] = fas_L13(0.01410,5,35,10,1,'Ar',0.9)
    %   >Fd = -5.2641e-09
    %   >Fc = 1.3605e-10
    %   >Fp = -6.0093e-10
    %   >Deq = 0.0014
    %   >k = 2.0377e-05
    %
    % DESCRIPTION:-------------------------------------------------------------
    %
    % Calculate air-sea fluxes and steady-state supersat based on:
    % Liang, J.-H., C. Deutsch, J. C. McWilliams, B. Baschek, P. P. Sullivan, 
    % and D. Chiba (2013), Parameterizing bubble-mediated air-sea gas exchange 
    % and its effect on ocean ventilation, Global Biogeochem. Cycles, 27, 
    % 894?905, doi:10.1002/gbc.20080.
    %
    % INPUTS:------------------------------------------------------------------
    % C:    gas concentration (mol m-3)
    % u10:  10 m wind speed (m/s)
    % SP:   Sea surface salinity (PSS)
    % pt:   Sea surface temperature (deg C)
    % pslp: sea level pressure (atm)
    % gas:  formula for gas (He, Ne, Ar, Kr, Xe, N2, or O2), formatted as a
    %       string, e.g. 'He'
    % rh:   relative humidity as a fraction of saturation (0.5 = 50% RH)
    %       rh is an optional but recommended argument. If not provided, it
    %       will be automatically set to 1 (100% RH).
    %
    %       Code    Gas name        Reference
    %       ----   ----------       -----------
    %       He      Helium          Weiss 1971
    %       Ne      Neon            Hamme and Emerson 2004
    %       Ar      Argon           Hamme and Emerson 2004
    %       Kr      Krypton         Weiss and Keiser 1978
    %       Xe      Xenon           Wood and Caputi 1966
    %       N2      Nitrogen        Hamme and Emerson 2004   
    %       O2      Oxygen          Garcia and Gordon 1992  
    %
    % OUTPUTS:-----------------------------------------------------------------
    %
    % tuple output: (Fd,Fc,Fp,Deq,k)
    % Fd:   Surface gas flux                              (mmol m-2 s-1)
    % Fc:   Flux from fully collapsing small bubbles      (mmol m-2 s-1)
    % Fp:   Flux from partially collapsing large bubbles  (mmol m-2 s-1)
    % Deq:  Equilibrium supersaturation                   (unitless (%sat/100))
    % k:    Diffusive gas transfer velocity               (m s-1)
    %
    % Note: Total air-sea flux is Ft = Fd + Fc + Fp
    %
    % REFERENCE:---------------------------------------------------------------
    %
    % Liang, J.-H., C. Deutsch, J. C. McWilliams, B. Baschek, P. P. Sullivan, 
    %   and D. Chiba (2013), Parameterizing bubble-mediated air-sea gas 
    %   exchange and its effect on ocean ventilation, Global Biogeochem. Cycles, 
    %   27, 894?905, doi:10.1002/gbc.20080.
    %
    % AUTHOR:---------------------------------------------------------------
    % Written by David Nicholson dnicholson@whoi.edu
    % Modified by Cara Manning cmanning@whoi.edu
    % Woods Hole Oceanographic Institution
    % Version: 12 April 2017
    %
    % COPYRIGHT:---------------------------------------------------------------
    %
    % Copyright 2017 David Nicholson and Cara Manning 
    %
    % Licensed under the Apache License, Version 2.0 (the "License");
    % you may not use this file except in compliance with the License, which 
    % is available at http://www.apache.org/licenses/LICENSE-2.0
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    # -------------------------------------------------------------------------
    # Conversion factors
    # -------------------------------------------------------------------------
    m2cm = 100 # cm in a meter
    h2s = 3600 # sec in hour
    
    # -------------------------------------------------------------------------
    # Calculate water vapor pressure and adjust sea level pressure
    # -------------------------------------------------------------------------
    
    # if humidity is not provided, set to 1 for all values   
    ph2oveq = vpress_sw(SP,pt)
    ph2ov = rh * ph2oveq
    
    # slpc = (observed dry air pressure)/(reference dry air pressure)   
    slp_corr = (slp - ph2ov) /(1 - ph2oveq)
    
    # -------------------------------------------------------------------------
    # Parameters for COARE 3.0 calculation
    # -------------------------------------------------------------------------
    
    # Calculate potential density at surface
    SA = SP * 35.16504 / 35
    CT = CT_from_pt(SA,pt)
    rhow = rho(SA,CT,0)
    rhoa = 1.225
    
    lam = 13.3
    A = 1.3
    phi = 1
    tkt = 0.01
    hw = lam /A / phi
    ha = lam 
    
    # air-side schmidt number
    ScA = 0.9
       
    # -------------------------------------------------------------------------
    # Calculate gas physical properties
    # -------------------------------------------------------------------------
    
    if chi_atm == None:
        xG = air_mol_fract(gas=gas)
        Geq = eq_SP_pt(SP,pt,gas=gas,units="mM")
    else:
        xG = chi_atm
        Geq = xG * sol_SP_pt(SP,pt,gas=gas,units="mM")
        
    alc = (Geq / atm2pa) * R * (pt+K0)
    
    Gsat = C / Geq
    ScW = schmidt(SP,pt,gas=gas)
    
    # -------------------------------------------------------------------------
    # Calculate COARE 3.0 and gas transfer velocities
    # -------------------------------------------------------------------------
    # ustar
    cd10 = cdlp81(u10)
    ustar = u10 * np.sqrt(cd10)
    
    # water-side ustar
    ustarw = ustar / np.sqrt(rhow / rhoa)
    
    # water-side resistance to transfer
    rwt = np.sqrt(rhow / rhoa) * (hw * np.sqrt(ScW)+(np.log(0.5 / tkt) /0.4))
    
    # air-side resistance to transfer
    rat = ha * np.sqrt(ScA) + 1 / np.sqrt(cd10) - 5 + 0.5 * np.log(ScA) /0.4
    
    # diffusive gas transfer coefficient (L13 eqn 9)
    Ks = ustar / (rwt + rat * alc)
    
    # bubble transfer velocity (L13 eqn 14)
    Kb = 1.98e6 * ustarw**2.76 * (ScW / 660)**(-2/3) / (m2cm * h2s)
    
    # overpressure dependence on windspeed (L13 eqn 16)
    dP = 1.5244 * ustarw**1.06 
      
    # -------------------------------------------------------------------------
    # Calculate air-sea fluxes
    # -------------------------------------------------------------------------
    
    Fd = Ks * Geq * (slp_corr - Gsat) # Fs in L13 eqn 3
    Fp = Kb * Geq * ((1+dP) * slp_corr - Gsat) # Fp in L13 eqn 3
    Fc = xG * 5.56 * ustarw ** 3.86 # L13 eqn 15
    
    # -------------------------------------------------------------------------
    # Calculate steady-state supersaturation 
    # -------------------------------------------------------------------------
    # L13 eqn 5
    Deq = (Kb * Geq * dP * slp_corr + Fc) / ((Kb + Ks) * Geq * slp_corr) 
    return (Fd,Fc,Fp,Deq,Ks)
    
@match_args_return
def N16(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None,Ainj=):
    """
    % Function to calculate air-sea gas exchange flux using Nicholson 16
    % parameterization
    %
    % USAGE:-------------------------------------------------------------------
    %  
    % [Fd, Fc, Fp, Deq, k] = fas_N11(C,u10,S,T,slp,gas,rh)
    % [Fd, Fc, Fp, Deq, k] = fas_N11(0.01410,5,35,10,1,'Ar',0.9)
    %
    % Fd = -4.4860e-09
    % Fc = 1.9911e-10
    % Fp = 2.5989e-11
    % Deq = 9.3761e-04
    % k = 1.7365e-05
    %
    % DESCRIPTION:-------------------------------------------------------------
    %
    % Calculate air-sea fluxes and steady-state supersaturation based on:
    % Nicholson, D. P., S. Khatiwala, and P. Heimbach (2016), Noble gas tracers
    %   of ventilation during deep-water formation in the Weddell Sea, 
    %   IOP Conf. Ser. Earth Environ. Sci., 35(1), 012019, 
    %   doi:10.1088/1755-1315/35/1/012019.
    %
    % which updates Ainj and Aex parameters from:
    %
    % Nicholson, D., S. Emerson, S. Khatiwala, R. C. Hamme. (2011) 
    %   An inverse approach to estimate bubble-mediated air-sea gas flux from 
    %   inert gas measurements.  Proceedings on the 6th International Symposium
    %   on Gas Transfer at Water Surfaces.  Kyoto University Press.
    %
    % Fc = Ainj * slpc * Xg * u3
    % Fp = Aex * slpc * Geq * D^n * u3
    %
    % where u3 = (u-2.27)^3 (and zero for  u < 2.27)
    %
    % Explanation of slpc:
    %      slpc = (observed dry air pressure)/(reference dry air pressure)
    % slpc is a pressure correction factor to convert from reference to
    % observed conditions. Equilibrium gas concentration in gasmoleq is
    % referenced to 1 atm total air pressure, including saturated water vapor
    % (RH=1), but observed sea level pressure is usually different from 1 atm,
    % and humidity in the marine boundary layer can be less than saturation.
    % Thus, the observed sea level pressure of each gas will usually be
    % different from the reference.
    %
    % INPUTS:------------------------------------------------------------------
    % 
    % C:    gas concentration in mmol L-1
    % u10:  10 m wind speed (m/s)
    % S:    Sea surface salinity (PSS)
    % T:    Sea surface temperature (deg C)
    % slp:  sea level pressure (atm)
    %
    % gas:  formula for gas, formatted as a string, e.g. 'He'
    %       Code    Gas name        Reference
    %       ----   ----------       -----------
    %       He      Helium          Weiss 1971
    %       Ne      Neon            Hamme and Emerson 2004
    %       Ar      Argon           Hamme and Emerson 2004
    %       Kr      Krypton         Weiss and Keiser 1978
    %       Xe      Xenon           Wood and Caputi 1966
    %       N2      Nitrogen        Hamme and Emerson 2004   
    %       O2      Oxygen          Garcia and Gordon 1992   
    %
    % 
    % rhum: relative humidity as a fraction of saturation (0.5 = 50% RH).
    %             default = 1 (100% RH)
    % chi_atm: atmospheric dry mixing ratio [atm] not needed for gases with
    %           known, constant mixing ratio
    %
    % OUTPUTS:-----------------------------------------------------------------
    % Fd:   Surface air-sea diffusive flux based on 
    %       Sweeney et al. 2007                           [mol m-2 s-1]
    % Fc:   Injection bubble flux (complete trapping)     [mol m-2 s-1]
    % Fp:   Exchange bubble flux (partial trapping)       [mol m-2 s-1]
    % Deq:  Steady-state supersaturation                  [unitless (%sat/100)]
    % k:    Diffusive gas transfer velocity               (m s-1)
    %
    % Note: Total air-sea flux is Ft = Fd + Fc + Fp
    %
    % REFERENCE:---------------------------------------------------------------
    % Nicholson, D. P., S. Khatiwala, and P. Heimbach (2016), Noble gas tracers
    %   of ventilation during deep-water formation in the Weddell Sea, 
    %   IOP Conf. Ser. Earth Environ. Sci., 35(1), 012019, 
    %   doi:10.1088/1755-1315/35/1/012019.
    %
    % AUTHORS:-----------------------------------------------------------------
    % David Nicholson dnicholson@whoi.edu
    % Cara Manning cmanning@whoi.edu
    % Woods Hole Oceanographic Institution
    % 
    %
    % COPYRIGHT:---------------------------------------------------------------
    %
    % Copyright 2017 David Nicholson and Cara Manning 
    %
    % Licensed under the Apache License, Version 2.0 (the "License");
    % you may not use this file except in compliance with the License, which 
    % is available at http://www.apache.org/licenses/LICENSE-2.0
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """   
    Ainj = 1.06e-9
    Aex = 2.19e-6
    
    ph2oveq = vpress_sw(SP,pt)
    ph2ov = rh * ph2oveq
    
    # slpc = (observed dry air pressure)/(reference dry air pressure)
    slp_corr = (slp - ph2ov) /(1 - ph2oveq)
    slpd = slp-ph2ov
    
    D = diff(SP,pt,gas=gas)
    Sc = schmidt(SP,pt,gas=gas)

    if chi_atm == None:
        xG = air_mol_fract(gas=gas)
        Ceq = eq_SP_pt(SP,pt,gas=gas,units="mM")
    else:
        xG = chi_atm
        Ceq = xG * sol_SP_pt(SP,pt,gas=gas,units="mM")
    
    # calculate wind speed term for bubble flux
    u3 = (u10 - 2.27)**3
    if isinstance(u3,float):
        u3 = np.asarray(u3)
    u3[u3 < 0] = 0
    
    k = kgas(u10,Sc,param='Sw07')
    Fd = k * (C - slp_corr * Ceq)
    Fc = -Ainj * slpd * xG * u3
    Fp = -Aex * slp_corr *Ceq * D**0.5 * u3
    Deq = -((Fp + Fc) / k) / Ceq
    return (Fd,Fc,Fp,Deq,k)

def N11(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None):
    """
    Same as N16 above but with:
        Ainj = 2.51e-9 / 1.5
        Aex = 1.15e-5 / 1.5
        
    % REFERENCE:---------------------------------------------------------------
    % Nicholson, D., S. Emerson, S. Khatiwala, R. C. Hamme (2011)
    %   An inverse approach to estimate bubble-mediated air-sea gas flux from 
    %   inert gas measurements.  Proceedings on the 6th International Symposium
    %   on Gas Transfer at Water Surfaces.  Kyoto University Press.
    """"
    
    # 1.5 factor converts from average winds to instantaneous - see N11 ref.
    Ainj = 2.51e-9 / 1.5
    Aex = 1.15e-5 / 1.5
    
    ph2oveq = vpress_sw(SP,pt)
    ph2ov = rh * ph2oveq
    
    # slpc = (observed dry air pressure)/(reference dry air pressure)
    slp_corr = (slp - ph2ov) /(1 - ph2oveq)
    slpd = slp-ph2ov
    
    D = diff(SP,pt,gas=gas)
    Sc = schmidt(SP,pt,gas=gas)

    if chi_atm == None:
        xG = air_mol_fract(gas=gas)
        Ceq = eq_SP_pt(SP,pt,gas=gas,units="mM")
    else:
        xG = chi_atm
        Ceq = xG * sol_SP_pt(SP,pt,gas=gas,units="mM")
    
    # calculate wind speed term for bubble flux
    u3 = (u10 - 2.27)**3
    if isinstance(u3,float):
        u3 = np.asarray(u3)
    u3[u3 < 0] = 0
    
    k = kgas(u10,Sc,param='Sw07')
    Fd = k * (C - slp_corr * Ceq)
    Fc = -Ainj * slpd * xG * u3
    Fp = -Aex * slp_corr *Ceq * D**0.5 * u3
    Deq = -((Fp + Fc) / k) / Ceq
    return (Fd,Fc,Fp,Deq,k)


    


    
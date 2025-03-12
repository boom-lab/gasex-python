#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:55:36 2017

@author: dnicholson
"""
import numpy as np
#from metpy.calc import density # use metpy to calculate air density for air-side Schmidt
#from metpy.units import units
from gasex.phys import vpress_sw, R, cdlp81, atm2pa, K0, xH2O_from_rh, calculate_air_density
from gasex.diff import schmidt,diff,air_side_Schmidt_number
from gasex.sol import sol_SP_pt,eq_SP_pt, air_mol_fract
from gasex.fugacity import fugacity_factor
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



def fsa(C,u10,SP,T,*,slp=1.0,gas=None,param="W14",rh=1,chi_atm=None):
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
    f  = fugacity_factor(T,gas=gas,slp=slp)
    C_eq = f * eq_SP_pt(SP,T,gas=gas,slp=slp,chi_atm=chi_atm) # apply fugacity factor outside of eq_SP_pt
    Sc = schmidt(SP,T,gas=gas)
    # piston velocity [m s-1]
    k = kgas(u10,Sc,param=param)
    Fd = k * (C - C_eq) # don't need slp_corr because it's now built into eq_SP_pt function * slp_corr)
    return Fd

def fsa_pC(pC_w,pC_a,u10,SP,T,*,gas=None,param="W14",chi_atm=None):
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
    s = sol_SP_pt(SP,T,chi_atm=chi_atm, gas=gas,units="mM")
    Sc = schmidt(SP,T,gas=gas)
    # piston velocity m s-1
    k = kgas(u10,Sc,param=param)
    # m s-1 * mol m-3 atm -1 * atm == mol m-2 s-1
    Fd = k * s * (pC_w - pC_a) / 1e6
    return Fd

@match_args_return
def L13(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None, air_temperature=None,
        calculate_schmidtair=True, schmidt_parameterization=None, return_vars=None,
        Ks = None, Kb=None, Kc=None, dP=None, beta=1.0, pressure_mode=False):
    """
    DESCRIPTION
    -------------
    Calculate air-sea fluxes and steady-state supersaturation using the Liang et al. (2013) parameterization.

    F = k (C - Ceq_slp)

    INPUTS:
    ----------
    C         Surface dissolved gas concentration       [mol m-3 == mmol L-1]
    u10       10-m wind speed                           [m s-1]
    SP        Practical Salinity                        --
    T         Surface water temperature                [deg C]
    slp       Sea-level pressure                       [atm]
    gas       Abbreviation for gas of interest         [e.g., 'He', 'Ar', 'N2O']
    rh        Fractional relative humidity             (1 = 100%)
    chi_atm   Atmospheric mixing ratio of the gas      [mol/mol]
    air_temperature  Air temperature                   [deg C] (optional; default air density = 1.225 kg/m³)
    calculate_schmidtair  Compute air-side Schmidt number (True/False; default = False)
    schmidt_parameterization  Method for water-side Schmidt number ('viscdiff', 'W14', or 'W92'; default = 'viscdiff')
    return_vars  Variables to return (default = ('Fd', 'Fc', 'Fp', 'Deq', 'Ks'))
    Ks        Diffusive gas transfer coefficient       [m s-1] (optional; overrides calculation if provided)
    Kb        Large bubble transfer velocity           [m s-1] (optional; overrides calculation if provided)
    Kc        Small bubble transfer velocity           [m s-1] (optional; overrides calculation if provided)
    dP        Overpressure dependence on windspeed     [%] (optional; overrides calculation if provided)
    beta      Scaling factor for bubble flux adjustment (default = 1)
    pressure_mode  If True, C is gas partial pressure in seawater [atm]; otherwise, C is concentration [mol m-3]

    OUTPUTS:
    ----------
    Fd        Surface gas flux                         [mol m-2 s-1]
    Fc        Flux from fully collapsing small bubbles [mol m-2 s-1]
    Fp        Flux from partially collapsing large bubbles [mol m-2 s-1]
    Deq       Equilibrium supersaturation             [unitless (%sat/100)]
    Ks        Diffusive gas transfer velocity         [m s-1]
    Kb        Large bubble transfer velocity          [m s-1]
    Kc        Small bubble transfer velocity          [m s-1]
    dP        Overpressure dependence on windspeed    [%]
    ScW       Water-side Schmidt number               [dimensionless]
    Geq       Equilibrium concentration at 1 atm      [mmol L-1]
    ustarw    Friction velocity                       [m s-1]
    rwt       Water-side resistance to transfer       [dimensionless]
    rat       Air-side resistance to transfer         [dimensionless]

    Note: Total air-sea flux is Ft = Fd + Fc + Fp

    REFERENCE:
    -------------
    Liang, J.-H., C. Deutsch, J. C. McWilliams, B. Baschek, P. P. Sullivan, and D. Chiba (2013), Parameterizing bubble-mediated air-sea gas exchange and its effect on ocean ventilation, Global Biogeochem. Cycles, 27, 894–905, doi:10.1002/gbc.20080.

    AUTHOR:
    -------------
    Written by David Nicholson
    dnicholson@whoi.edu
    Modified by Cara Manning
    cmanning@whoi.edu
    Modified by Colette Kelly
    colette.kelly@whoi.edu
    Woods Hole Oceanographic Institution
    Version: 12 March 2025

    COPYRIGHT:
    -------------
    Copyright 2017 David Nicholson and Cara Manning
    Licensed under the Apache License, Version 2.0
    http://www.apache.org/licenses/LICENSE-2.0
    """
    # -------------------------------------------------------------------------
    # Conversion factors
    # -------------------------------------------------------------------------
    m2cm = 100 # cm in a meter
    h2s = 3600 # sec in hour
    atm2pa = 101325.0 # Pa per atm, Pa/atm, 1 Pa = kg/m/s2
    R = 8.3144598 # ideal gas constant, J/(mol K), 1 J = kg*m2/s2
    # -------------------------------------------------------------------------
    # Calculate water vapor pressure and adjust sea level pressure
    # -------------------------------------------------------------------------

    # if humidity is not provided, set to 1 for all values
    ph2oveq = vpress_sw(SP,pt) # atm
    ph2ov = rh * ph2oveq # atm

    # slpc = (observed dry air pressure)/(reference dry air pressure)
    slp_corr = (slp - ph2ov) /(1 - ph2oveq) # dimensionless

    # -------------------------------------------------------------------------
    # Calculate air density and Schmidt number if air temperature is not None
    # -------------------------------------------------------------------------
    if air_temperature is not None:
        # Convert air temperature to Kelvin
        T = (air_temperature + K0) # K

        # calculate mixing ratio of water vapor in air
        xH2O = xH2O_from_rh(SP,pt,slp=slp,rh=rh)

        # Calculate density of air
        rhoa = calculate_air_density(air_temperature, slp, xH2O)

        # air-side schmidt number, dimensionless
        ScA = air_side_Schmidt_number(air_temperature, rhoa, gas=gas, calculate=calculate_schmidtair)

    else:
        rhoa = 1.225 # default density of air, kg/m3
        ScA = 0.9 # default air Schmidt number, dimensionless
    # -------------------------------------------------------------------------
    # Parameters for COARE 3.0 calculation - diffusive gas exchange
    # -------------------------------------------------------------------------

    # Calculate potential density at surface
    SA = SP * 35.16504 / 35 # absolute salinity, psu
    CT = CT_from_pt(SA,pt) # conservative temperature, degrees C
    rhow = rho(SA,CT,0*CT) # density of water, kg/m3

    # Constants for COARE 3.0 calculation
    lam, A, phi, tkt, hw, ha, zw, za, kappa = 13.3, 1.3, 1, 0.01, 13.3/1.3, 13.3, 0.5, 1.0, 0.4

    # -------------------------------------------------------------------------
    # Calculate gas physical properties
    # -------------------------------------------------------------------------
    if chi_atm is None:
        xG = air_mol_fract(gas=gas)
        f  = fugacity_factor(pt,gas=gas,slp=slp) # calculate f here s.t. eq_SP_pt can be referenced to 1 atm
        s = sol_SP_pt(SP,pt,chi_atm=xG, gas=gas,units="mM")
        Geq = f * eq_SP_pt(SP,pt,slp = 1.0,gas=gas,units="mM") # referenced to 1 atm, we apply pressure correction later
        Patm = Geq / s
    else:
        xG = chi_atm
        f  = fugacity_factor(pt,gas=gas,slp=slp)
        s = sol_SP_pt(SP,pt,chi_atm=xG, gas=gas,units="mM")
        Patm = xG * f  * (1 - ph2oveq) # referenced to 1 atm, we apply pressure correction later
        Geq = Patm * s

    T = pt + 273.15 # K
    alc = (Geq / (slp * atm2pa)) * R * T # dividing by slp makes alc dimensionless

    if pressure_mode: # allows user to input partial pressure instead of concentration
        Psw = C
        Gsat = Psw / Patm
    else:
        Gsat = C / Geq # dimensionless

    ScW = schmidt(SP,pt,gas=gas,schmidt_parameterization=schmidt_parameterization) # dimensionless

    # -------------------------------------------------------------------------
    # Calculate COARE 3.0 and gas transfer velocities
    # -------------------------------------------------------------------------
    # ustar
    cd10 = cdlp81(u10) # L13 eqn. 13
    ustar = u10 * np.sqrt(cd10)
    
    # water-side ustar - L13 eqn. 12
    ustarw = ustar / np.sqrt(rhow / rhoa) # m/s

    # water-side resistance to transfer, dimensionless
    # eqn. 13 of Fairall et al., 2011
    rwt = np.sqrt(rhow / rhoa) * (hw * np.sqrt(ScW)+(np.log(zw / tkt) / kappa)) # Fairall et al. 2011 define k as 0.4

    # air-side resistance to transfer, dimensionless
    # eqn. 15 of Fairall et al., 2011
    rat = ha * np.sqrt(ScA) + za / np.sqrt(cd10) - 5 + 0.5 * np.log(ScA) / kappa

    # diffusive gas transfer coefficient (L13 eqn 9/Fairall 2011 eqn. 11)
    if Ks is None:
        Ks = ustar / (rwt + rat * alc) # m/s
    # bubble transfer velocity (L13 eqn 14)
    if Kb is None:
        Kb = 1.98e6 * ustarw**2.76 * (ScW / 660)**(-2/3) / (m2cm * h2s) # m/s
    # small bubble transfer coefficient (L13 eqn 15)
    if Kc is None:
        Kc = 5.56 * ustarw**3.86 # mol/m2/s
    # overpressure dependence on windspeed (L13 eqn 16)
    if dP is None:
        dP = 1.5244 * ustarw**1.06 # %

    # -------------------------------------------------------------------------
    # Calculate air-sea fluxes
    # -------------------------------------------------------------------------

    Fd = Ks * Geq * (slp_corr - Gsat) # Fs in L13 eqn 3, units are mol/m2/s
    Fp = beta * Kb * Geq * ((1+dP) * slp_corr - Gsat) # Fp in L13 eqn 3, units are mol/m2/s
    Fc = beta * xG * Kc # L13 eqn 15, units are mol/m2/s

    # -------------------------------------------------------------------------
    # Calculate steady-state supersaturation
    # -------------------------------------------------------------------------
    Deq = (Kb * Geq * dP * slp_corr + Fc) / ((Kb + Ks) * Geq * slp_corr) # L13 eqn 5, units are percentage

    # -------------------------------------------------------------------------
    # define which variables get returned
    # -------------------------------------------------------------------------
    # Return requested variables
    if return_vars is None:
        return_vars = ['Fd', 'Fc', 'Fp', 'Deq', 'Ks']

    if isinstance(return_vars, str):
        return_vars = [return_vars]

    # Dictionary of all variables
    all_vars = {
        'Fd': Fd,
        'Fc': Fc,
        'Fp': Fp,
        'Deq': Deq,
        'Ks': Ks,
        'Kb': Kb,
        'Kc': Kc,
        'dP': dP,
        'ScW': ScW,
        'Geq': Geq,
        'ustarw': ustarw,
        'rwt': rwt,
        'rat': rat
    }

    # Return only the requested variables
    return tuple(all_vars[var] for var in return_vars)

@match_args_return
def N16(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None):
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
    """
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

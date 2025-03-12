#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Diffusion coeff and Schmidt number for gases in fresh/sea water and air
%=========================================================================
% Modified by D. Nicholson from MATLAB gas_diffusion Version 2.0 16 July 2013
%          Author: Roberta C. Hamme (University of Victoria)
% Diffusion values for 'He','Ne','Ar','Kr','Xe','N2','O2','CH4','N2' and 'CO2' are calculated from
% gas_diffusion Version 2.0 functions 
% salinity correction is of the form: D = D0 * (1 - 0.049 * SP / 35.5)
% 
%
% Support for additional gases ('CO2','N2O','CH4','RN','SF6','DMS','CFC12','CFC11','CH3BR','CCL4')
% has been added based on Wanninkhof 2014.
%
% Table 1:
% Sc = A + Bt + Ct2+ dt3+ Et4(t in °C). The last column is the calculated Schmidt number for 20°C. 
% The Schmidt number is the kinematic viscosity of waterdivided by the molecular diffusion 
% coefficient of the gas. The kinematic viscosity for fresh water and seawater are from 
% Sharqawy et al. (2010). The dif-fusion coefficients of gases are from the following: 
% 3He, He, Ne, Kr, Xe, CH4, CO2, and Rn measured by Jähne et al. (1987); Ar, O2, N2, N2O, 
% and CCl4fitusing Wilke and Chang (1955) as adapted by Hayduk and Laudie (1974); SF6 
% measured by King and Saltzman (1995); DMS measured by Saltzman etal. (1993); CFC-11 and 
% CFC-12 measured by Zheng et al. (1998); CH3Br measured by De Bruyn and Saltzman (1997a).
%
%
% REFERENCE:
%    He, Ne, Kr, Xe, CH4, CO2, H2 freshwater values from Jahne et al., 1987.
%       "Measurement of Diffusion Coeffients of Sparingly Soluble Gases in Water"
%       J. Geophys. Res., 92(C10), 10767-10776.
%    Ar freshwaters values are extrapolated from Jahne et al. 1987
%       He, Ne, Kr, Xe values at each temperature were fitted to D vs. mass^-0.5
%       relationship to predict Ar at those temperatures, then Ar was fit to a
%       ln(D_Ar) vs. 1/T(K) relationship to obtain Eyring equation coefficients
%    O2 and N2 freshwater values from  Ferrell and Himmelblau, 1967.
%       "Diffusion coefficients of nitrogen and oxygen in water"
%       J. Chem. Eng. Data, 12(1), 111-115, doi: 10.1021/je60032a036.
%    Correction for salinity is based on Jahne's observed average 4.9% decrease in
%       diffusivity for H2 and He in 35.5 ppt NaCl solution
%
%    for Ne, the Jahne values compare well with and fall between those of
%       Wise and Houghton 1968 and Holz et al. 1994
%    for Ar, the extrapolated Jahne values compare well with Wise and Houghton 1968,
%       O'Brien and Hyslop 1977, and a numerical simulation by Bourg et al. 2008
%       but are higher than other reported values
%    for Kr, the Jahne values compare well with Wise and Houghton 1968,
%       and a numerical simulation by Bourg et al. 2008
%    for Xe, the Jahne values compare well with Pollack 1981, and a numerical
%       simulation by Bourg et al. 2008, but fall significantly above Wise and Houghton 1968
%       and below Weingartner et al. 1992
%    for O2, there is general agreement among measurements. The Ferrel and Himmelblau values
%       agree reasonably well with Baird and Davidson 1962, Wise and Houghton 1966,
%       Duda and Vrentas 1968, O'Brien and Hyslop 1977, and the Wilke and Change (1955) theory
%       as tabulated by Wanninkhof 1992, but lie below Krieger et al 1967
%    for N2, there is less agreement. The Ferrel and Himmelblau values
%       agree reasonably well with Baird and Davidson 1962, O'Brien and Hyslop 1977,
%       and the Wilke and Change (1955) theory as tabulated by Wanninkhof 1992,
%       but lie significantly below the values of Wise and Houghton 1966 and Krieger et al 1967
%    for He, I did not investigate comparisons of data, but chose Jahne
%       since their work for other gases appears to be the best
%    for CO2, CH4 and H2: Jahne 1987 
%
%   
%
% DISCLAIMER:
%    This software is provided "as is" without warranty of any kind.
%=========================================================================
"""
from __future__ import division
import numpy as np
from numpy.polynomial.polynomial import polyval
from ._utilities import match_args_return
from gasex.phys import kinematic_viscosity_air
from gasex.phys import R as R 
from gasex.phys import visc as visc
from gasex.phys import kinematic_viscosity_air


# Currently supported gases
# TODO: find N2O, CO diffusivities
GAS_LIST = ('HE','NE','AR','KR','XE','N2','O2','CH4','N2','CO2')
W14_LIST = ('CO2','N2O','CH4','RN','SF6','DMS','CFC12','CFC11','CH3BR','CCL4')
W92_LIST = ('HE','NE','AR','O2','CH4','CO2','N2','KR','N2O','RN','SF6','CCL2','CCL3')

@match_args_return
def diff(SP,pt,*,gas=None):
    
    """
    DESCRIPTION
    -----------
       Diffusion coefficients of various gases in fresh/sea water
    
    PARAMETERS
    -----------
      SP = practical salinity       [PSS-78]
      pt = potential temperature    [degree C]
      gas = 'He','Ne','Ar','Kr','Xe','N2','O2','CH4','N2' or 'CO2'
    
    OUTPUT:
      D = diffusion coefficient     [m^2 s-1]

    """
    g_up = gas.upper()
    if g_up not in GAS_LIST:
        raise ValueError("gas: must be one of ", GAS_LIST)
        
    AEa_dict = {'O2': (4.286e-6, 18700),\
           'HE': (0.8180e-6, 11700),\
           'NE': (1.6080e-6, 14840),\
           'AR': (2.227e-6, 16680),\
           'KR': (6.3930e-6, 20200),\
           'XE': (9.0070e-6, 21610),\
           'N2': (3.4120e-6, 18500),\
           'CH4':(3.0470e-6, 18360),\
           'CO2':(5.0190e-6, 19510),\
           'H2': (3.3380e-6, 16060)}
           
    if g_up in AEa_dict.keys():
        #freshwater diffusivity
        AEa = AEa_dict[g_up]
        D0 = AEa[0] * np.exp(-AEa[1] / (R * (pt+273.15)))
        #salinity correction
        D = D0 * (1 - 0.049 * SP / 35.5)
    else:
        raise ValueError("gas: must be one of ", AEa_dict.keys())
    return D


@match_args_return
def schmidt(SP,pt,*,gas=None, schmidt_parameterization=None):
    """
    DESCRIPTION
    -------------
    Calculate water-side Schmidt number from salinity and potential temperature.
    
    Schmidt number (Sc) is a dimensionless number representing the ratio of momentum diffusivity 
    (kinematic viscosity) to mass diffusivity. It is a key parameter in air-sea gas exchange calculations.
    
    INPUTS:
    ----------
    SP       Practical salinity                            [dimensionless]
    pt       Potential temperature                        [deg C]
    gas      Abbreviation for gas of interest             [string]
             ('NE', 'CFC11', 'HE', 'CH4', 'SF6', 'XE', 'CCL2', 'CO2', 
              'CFC12', 'DMS', 'AR', 'RN', 'CCL4', 'KR', 'O2', 'N2', 
              'CCL3', 'CH3BR', or 'N2O')
    schmidt_parameterization  Method for Schmidt number calculation [string]
             ('viscdiff', 'W14', or 'W92')
             Default: 'viscdiff' if available for that gas,
                      'W14' if viscdiff is not available,
                      'W92' if neither viscdiff nor W14 are available.
    
    OUTPUT:
    ----------
    schmidt  Water-side Schmidt number (dimensionless)
    
    AUTHOR: David Nicholson & Colette Kelly
    """
    GAS_LIST = ('HE','NE','AR','KR','XE','N2','O2','CH4','N2','CO2')
    W14_LIST = ('CO2','N2O','CH4','RN','SF6','DMS','CFC12','CFC11','CH3BR','CCL4')
    W92_LIST = ('HE','NE','AR','O2','CH4','CO2','N2','KR','N2O','RN','SF6','CCL2','CCL3')
    
    g_up = gas.upper()
    
    if schmidt_parameterization is None:
        try:
            if g_up in GAS_LIST:
                return visc(SP, pt) / diff(SP, pt, gas=gas)
            elif g_up in W14_LIST:
                return schmidt_W14(pt, gas=gas, sw=True)
            elif g_up in W92_LIST:
                return schmidt_W92(pt, gas=gas, sw=True)
            else:
                raise ValueError(f"gas {g_up} does not match one of {set(GAS_LIST + W14_LIST + W92_LIST)}")
        except ValueError:
            raise ValueError(f"gas {g_up} does not match one of {set(GAS_LIST + W14_LIST + W92_LIST)}")
    
    if g_up:
        if g_up not in GAS_LIST and g_up not in W14_LIST and g_up not in W92_LIST:
            raise ValueError(f"gas {g_up} does not match one of {set(GAS_LIST + W14_LIST + W92_LIST)}")
        
        if g_up in GAS_LIST and schmidt_parameterization == "viscdiff":
            return visc(SP, pt) / diff(SP, pt, gas=gas)
        
        if g_up in W14_LIST and schmidt_parameterization == "W14":
            return schmidt_W14(pt, gas=gas, sw=True)
        
        if g_up in W92_LIST and schmidt_parameterization == "W92":
            return schmidt_W92(pt, gas=gas, sw=True)
        
        if schmidt_parameterization == "viscdiff":
            return visc(SP, pt) / diff(SP, pt, gas=gas)
        
        if schmidt_parameterization == "W14":
            return schmidt_W14(pt, gas=gas, sw=True)
        
        if schmidt_parameterization == "W92":
            return schmidt_W92(pt, gas=gas, sw=True)

@match_args_return
def schmidt_W14(pt,*,gas=None,sw=True):
    """
    DESCRIPTION
    -------------
    Calculate the Schmidt number at 35 PSU or for fresh water based on Wanninkhof 2014 Table 1.
    
    Schmidt number (Sc) is a dimensionless quantity representing the ratio of momentum diffusivity 
    (kinematic viscosity) to mass diffusivity. It is essential in air-sea gas exchange calculations.

    INPUTS:
    ----------
    pt       Potential temperature                        [deg C]
    gas      Abbreviation for gas of interest             [string]
             ('CO2', 'N2O', 'CH4', 'RN', 'SF6', 'DMS', 'CFC12', 
              'CFC11', 'CH3BR', or 'CCL4')
    sw       Seawater flag                                [bool]
             If True, calculates for seawater at SP = 35.
             If False, calculates for fresh water.
             Default: True.
    
    OUTPUT:
    ----------
    schmidt  Water-side Schmidt number (dimensionless)
    
    AUTHOR: David Nicholson
    """
    W14_LIST = ('CO2','N2O','CH4','RN','SF6','DMS','CFC12','CFC11','CH3BR','CCL4')
    g_up = gas.upper()
    if sw:
        A_dict = {'CO2': (2116.8,-136.25,4.7353,-0.092307,0.0007555 ),\
            'N2O': (2356.2,-166.38,6.3952,-0.13422,0.0011506 ),\
            'CH4':(2101.2,-131.54,4.4931,-0.08676,0.00070663),
            'RN': (3489.6,-244.56,8.9713,-0.18022,0.0014985 ),
            'SF6':(3177.5,-200.57,6.8865,-0.13335,0.0010877 ),
            'DMS':(2855.7,-177.63,6.0438,-0.11645,0.00094743),
            'CFC12':(3828.1,-249.86, 8.7603, -0.1716, 0.001408 ),
            'CFC11':(3579.2, -222.63, 7.5749, -0.14595, 0.0011874 ),
            'CH3BR':(2181.8, -138.4, 4.7663, -0.092448, 0.0007547 ),
            'CCL4': (4398.7, -308.25, 11.798, -0.24709, 0.0021159) }
    else:
        A_dict = {'CO2': (1923.6, -125.06, 4.3773, -0.085681, 0.00070284 ),\
            'N2O': (2141.2, -152.56, 5.8963, -0.12411, 0.0010655 ),\
            'CH4':(1909.4, -120.78, 4.1555, -0.080578, 0.00065777),
            'RN': (3171, -224.28, 8.2809, -0.16699, 0.0013915 ),
            'SF6':(3035, -196.35, 6.851, -0.13387, 0.0010972 ),
            'DMS':(2595, -163.12, 5.5902, -0.10817, 0.00088204),
            'CFC12':(3478.6, -229.32, 8.0961, -0.15923, 0.0013095 ),
            'CFC11':(3460, -217.49, 7.4537, -0.14423, 0.0011761 ),
            'CH3BR':(2109.2, -135.17, 4.6884, -0.091317, 0.00074715 ),
            'CCL4': (3997.2, -282.69, 10.88, -0.22855, 0.0019605) }

    if g_up in A_dict.keys():
        A = A_dict[g_up]
    else:
        raise ValueError("gas", g_up, " does not match one of ", A_dict.keys())

    Sc = polyval(pt,A)
    return Sc

@match_args_return
def schmidt_W92(pt,*,gas=None,sw=True):
    """
    DESCRIPTION
    -------------
    Calculate the Schmidt number at 35 PSU or for fresh water based on Wanninkhof 1992 Table A1.
    
    Schmidt number (Sc) is a dimensionless quantity representing the ratio of momentum diffusivity 
    (kinematic viscosity) to mass diffusivity. It is essential in air-sea gas exchange calculations.

    INPUTS:
    ----------
    pt       Potential temperature                        [deg C]
    gas      Abbreviation for gas of interest             [string]
             ('O2', 'N2', 'CO2', 'N2O', 'CH4', 'HE', 'AR', 'KR', 'XE', 'SF6', 'CCL2', 'CCL3', 'CCL4')
    sw       Seawater flag                                [bool]
             If True, calculates for seawater at SP = 35.
             If False, calculates for fresh water.
             Default: True.
    
    OUTPUT:
    ----------
    schmidt  Water-side Schmidt number (dimensionless)
    
    AUTHOR: David Nicholson
    """
    W92_LIST = ('HE','NE','AR','O2','CH4','CO2','N2','KR','N2O','RN','SF6','CCL2','CCL3')
    if gas is None:
        raise ValueError(f"please specify gas")
    else:
        g_up = gas.upper()

    A_dict = {
        'HE': (410.14, 20.503, 0.53175, 0.0060111),
        'NE': (855.1, 46.299, 1.254, 0.01449),
        'AR': (1909.1, 125.09, 3.9012, 0.048953),
        'O2': (1953.4, 128.00, 3.9918, 0.050091),
        'CH4': (2039.2, 120.31, 3.4209, 0.040437),
        'CO2': (2073.1, 125.62, 3.6276, 0.043219),
        'N2': (2206.1, 144.86, 4.5413, 0.056988),
        'KR': (2205.0, 135.71, 3.9549, 0.047339),
        'N2O': (2301.1, 151.1, 4.7364, 0.059431),
        'RN': (3412.8, 224.30, 6.7954, 0.08300),
        'SF6': (3531.6, 231.40, 7.2168, 0.090558),
        'CCL2': (3713.2, 243.30, 7.5879, 0.095215),
        'CCL3': (4039.8, 264.70, 8.2552, 0.10359)
    } if sw else {
        'HE': (377.09, 19.154, 0.50137, 0.005669),
        'NE': (764, 42.234, 1.1581, 0.013405),
        'AR': (1759.7, 117.37, 3.6959, 0.046527),
        'O2': (1800.6, 120.10, 3.7818, 0.047608),
        'CH4': (1897.8, 114.28, 3.2902, 0.039061),
        'CO2': (1911.1, 118.11, 3.4527, 0.041320),
        'N2': (1970.7, 131.45, 4.1390, 0.052106),
        'KR': (2032.7, 127.55, 3.7621, 0.045236),
        'N2O': (2055.6, 137.11, 4.3173, 0.054350),
        'RN': (3146.1, 210.48, 6.4486, 0.079135),
        'SF6': (3255.3, 217.13, 6.8370, 0.086070),
        'CCL2': (3422.7, 228.30, 7.1886, 0.090496),
        'CCL3': (3723.7, 248.37, 7.8208, 0.098455)
    }

    if g_up in A_dict.keys():
        A = A_dict[g_up]
    else:
        raise ValueError(f"gas {g_up} does not match one of {W92_LIST}")

    Sc = A[0] - A[1]*pt + A[2]*pt**2 - A[3]*pt**3
    return Sc

@match_args_return
def air_side_Schmidt_number(air_temperature, air_density, gas=None, calculate=True):
    """
    DESCRIPTION
    -------------
    Calculate the air-side Schmidt number.

    The Schmidt number (Sc) is a dimensionless quantity representing the ratio of momentum diffusivity 
    (kinematic viscosity) to mass diffusivity. It is important for characterizing gas exchange processes 
    between air and water.

    INPUTS:
    ----------
    air_temperature  Air temperature                        [deg C]
    air_density      Air density                            [kg m⁻³]
    rh              Relative humidity                      [dimensionless]
                    Default: 1 (100% humidity)
    gas             Abbreviation for gas of interest       [string]
                    ('H2O', 'CO2', 'CH4', 'CO', 'SO2', 'O3', 'NH3', or 'N2O')
    calculate       Schmidt number calculation method      [bool]
                    If True, calculates from diffusivity and kinematic viscosity of air.
                    If False, uses default value from a lookup table.

    OUTPUT:
    ----------
    schmidt         Air-side Schmidt number (dimensionless)

    AUTHOR: Colette Kelly
    """
    # Coefficients of diffusivity (cm2/s) of selected gases in air, 
    diffusivities = {
        'H2O': 0.2178, # Massman 1998
        'CO2': 0.1381, # Massman 1998
        'CH4': 0.1952, # Massman 1998
        'CO': 0.1807, # Massman 1998
        'SO2': 0.1089, # Massman 1998
        'O3': 0.1444, # Massman 1998
        'NH3': 0.1978, # Massman 1998
        'N2O': 0.1436, # Massman 1998
    }
    
    # empirical values from de Richter et al., 2017
    schmidt_dict = {
        'H2O': 0.61,
        'CH4': 0.69,
        'N2O': 0.93,
    }

    # can calculate Schmidt number for gases that we have the diffusivities for
    if (gas in diffusivities.keys())&(calculate==True):
        # Convert air temperature to Kelvin
        T = air_temperature + 273.15
        # Constants for N2O
        molecular_diffusivity = diffusivities[gas]*1e-4  # convert from cm^2/s to m2/s
        # Calculate air-side Schmidt number
        mu_a, nu_a = kinematic_viscosity_air(air_temperature, air_density)  # kg/m/s, m2/s
        ScA = nu_a / molecular_diffusivity # dimensionless
    # or look up an empirical value
    elif (gas in schmidt_dict.keys())&(calculate==False):
        ScA = schmidt_dict[gas]
    # or just use default value of 0.9 for all other cases
    else:
        ScA = 0.9
    
    return ScA

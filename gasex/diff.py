#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Diffusion coeff and Schmidt number for gases in fresh/sea water
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
from gasex.phys import R as R 
from gasex.phys import visc as visc


# Currently supported gases
# TODO: find N2O, CO diffusivities
GAS_LIST = ('HE','NE','AR','KR','XE','N2','O2','CH4','N2','CO2')

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
def schmidt(SP,pt,*,gas=None):

    g_up = gas.upper()
    if g_up not in GAS_LIST:
        raise ValueError("gas", g_up, " does not match one of ", GAS_LIST)
        
    Sc = visc(SP,pt) / diff(SP,pt,gas=gas) 
    return Sc

@match_args_return
def schmidt_W14(pt,*,gas=None,sw=True):
    """Schmidt number @ 35 psu based on Wanninkhof 2014 Table 1

    Args:
        pt ([array like]): potential temperature  [degree C]
        gas ([string]): abbreviation for gas. Defaults to None.
        sw (bool, optional): if True, then calculates for SP = 35, of false, 
            calculates for fresh water. Defaults to True.

    Raises:
        ValueError: [description]

    Returns:
        [type]: Schmidt number [dimensionless]
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

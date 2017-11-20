#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Diffusion coeff and Schmidt number for gases in fresh/sea water
%=========================================================================
% Modified by D. Nicholson from MATLAB gas_diffusion Version 2.0 16 July 2013
%          Author: Roberta C. Hamme (University of Victoria)
%
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
% DISCLAIMER:
%    This software is provided "as is" without warranty of any kind.
%=========================================================================
"""
from __future__ import division
import numpy as np
from gsw import CT_from_pt,rho
from ._utilities import match_args_return
from gasex.phys import R as R 
from gasex.phys import visc as visc



GAS_LIST = ['He','Ne','Ar','Kr','Xe','N2','O2','CH4','N2','CO2']

@match_args_return
def diff(SP,pt,*,gas=None):
    
    """
    DESCRIPTION
    -----------
       Diffusion coefficients of various gases in fresh/sea water
    
    PARAMETERS
    -----------
      SP = practical salinity       [PSS-78]
      pt = temperature               [degree C]
      gas = 'He','Ne','Ar','Kr','Xe','N2','O2','CH4','N2' or 'CO2'
    
    OUTPUT:
      D = diffusion coefficient     [m^2 s-1]

    """
    if gas not in GAS_LIST:
        raise ValueError("gas: must be one of ", GAS_LIST)
        
    AEa_dict = {'O2': [4.286e-6, 18700],\
           'He': [0.8180e-6, 11700],\
           'Ne': [1.6080e-6, 14840],\
           'Ar': [2.227e-6, 16680],\
           'Kr': [6.3930e-6, 20200],\
           'Xe': [9.0070e-6, 21610],\
           'N2': [3.4120e-6, 18500],\
           'CH4':[3.0470e-6, 18360],\
           'CO2':[5.0190e-6, 19510],\
           'H2': [3.3380e-6, 16060]}
           
    if gas in AEa_dict.keys():
        #freshwater diffusivity
        AEa = AEa_dict[gas]
        D0 = AEa[0] * np.exp(-AEa[1] / (R * (pt+273.15)))
        #salinity correction
        D = D0 * (1 - 0.049 * SP / 35.5)
    elif gas == 'CO2':
        #CO2 schmidt# formula from Wanninkhof (1992)
        D = visc(SP,pt) / schmidt(SP,pt,'CO2')
    return D


@match_args_return
def schmidt(SP,pt,*,gas=None):
    if gas not in GAS_LIST:
        raise ValueError("gas: must be one of ", GAS_LIST)
        
    Sc = visc(SP,pt) / diff(SP,pt,gas=gas) 
    return Sc


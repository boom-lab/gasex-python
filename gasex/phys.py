#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:02:39 2017

Common physical constants and properties for gas functions

@author: dnicholson
"""
from __future__ import division
import numpy as np
from gsw import rho, CT_from_pt
from ._utilities import match_args_return

# 0 Celsius in Kelvin
K0 = 273.15

# Gas Constant in J / mol K
R = 8.3144598

# Gas constant in L atm / (mol K)
RatmLmolK = 0.082057366080960

# 1 atm in Pa
atm2pa = 101325.0

# 1 atm in mmHg
atm2mmhg = 760.0

@match_args_return
def visc(SP,pt):
    """
    Calculated the Kinematic Viscosity of Seawater as a function of salinity
    Temperature

    Parameters
    ----------
    SP : array-like
        Practical Salinity
    pt : array-like
        Potential Temperature,      [degrees C]

    Returns
    -------
    visc : array-like,
        Kinematic Viscosity in      [m2 s-1]

    Author: David Nicholson
    -------
    Adapted from SW_VISC MATLAB function by Ayal Anis as described below
    %
    % visc
    %
    %
    % SW_VISC  $Revision: 0.0 $  $Date: 1998/01/19 $
    %          Copyright (C) Ayal Anis 1998.
    %
    % USAGE:  visc = sw_visc(S,T,P)
    %
    % DESCRIPTION:
    %    Calculates kinematic viscosity of sea-water.
    %    based on Dan Kelley's fit to Knauss's TABLE II-8
    """


    SA = SP * 35.16504/35
    CT = CT_from_pt(SA,pt)
    dens = rho(SA,CT,0*CT)
    visc = 1e-4 * (17.91 - 0.5381 * pt + 0.00694 * pt**2 + 0.02305 * SP) / dens
    return visc


@match_args_return
def vpress_sw(SP,pt):
    molal = 31.998 * SP / (1e3 - 1.005*SP)
    osmotic_coeff = 0.90799 -0.08992*(0.5*molal) + 0.18458*(0.5*molal)**2 - \
        0.07395*(0.5*molal)**3 - 0.00221*(0.5*molal)**4
    vpress_sw = vpress_w(pt) * np.exp(-0.018 * osmotic_coeff * molal)
    return vpress_sw

@match_args_return
def vpress_w(t):
    tmod = 1- (t + K0) / 647.096
    # Calculate value of Wagner polynomial
    wagner = -7.85951783*tmod + 1.84408259*tmod**1.5 - 11.7866497*tmod**3 + \
        22.6807411 * tmod**3.5 - 15.9618719*tmod**4 + 1.80122502*tmod**7.5
    # Vapor pressure of pure water in Pascals
    vpress_w = np.exp(wagner * 647.096 / (t + K0)) * 22.064 * 1e6 / atm2pa

    return vpress_w

@match_args_return
def cdlp81( u10):
    # Calculates drag coefficient from u10, wind speed at 10 m height

    cd = 4.9e-4 + 6.5e-5 * u10
    if isinstance(cd,float):
        cd = np.asarray(cd)
    cd[u10 <= 11] = 0.0012
    cd[u10 >= 20] = 0.0018
    return cd


@match_args_return
def u_2_u10(u_meas,height):
    """
    % u10 = calc_u10(umeas,hmeas)
    %
    % USAGE:-------------------------------------------------------------------
    %
    % [u10] = u_2_u10(5,4)
    %
    % >u10 = 5.5302
    %
    % DESCRIPTION:-------------------------------------------------------------
    % Scale wind speed from measurement height to 10 m height
    %
    % INPUTS:------------------------------------------------------------------
    % umeas:  measured wind speed (m/s)
    % hmeas:  height of measurement (m)
    %
    % OUTPUTS:-----------------------------------------------------------------
    %
    % u10:   calculated wind speed at 10 m
    %
    % REFERENCE:---------------------------------------------------------------
    %
    % Hsu S, Meindl E A and Gilhousen D B (1994) Determining the Power-Law
    %    Wind-Profile Exponent under Near-Neutral Stability Conditions at Sea
    %    J. Appl. Meteor. 33 757-765
    %
    % AUTHOR:---------------------------------------------------------------
    % David Nicholson -  Adapted from calc_u10.m by Cara Manning
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    u10 = u_meas * (10.0 / height)**0.11
    return u10

#@match_args_return
#def vpress_sw(SP,pt):
#    """
#    pv,w data from Ambrose and Lawrenson [88]
#    Validity: pv,sw and pv,w in (mm Hg); 0 < t68 < 40 oC; 0 < SP < 40 g/kg;
#    Accuracy: ±0.02%
#
#    REFERENCE:
#        F. J. Millero, The thermodynamics of seawater. Part II; Thermochemical
#        properties, Ocean Science and Engineering, 8(1), 1-40, 1983.
#
#        Eq. 33 in Sharqawy, Mostafa H., John H. Lienhard V and Syed M. Zubair.
#        "The thermophysical properties of seawater: A review of existing
#        correlations and data." Desalination and Water Treatment, 16
#        (April 2010) 354–380
#
#    """
#
#    t68 = pt * 1.00024     # pt68 is the potential temperature in degress C on
#              # the 1968 International Practical Temperature Scale IPTS-68.
#    A = -2.3311e-3 - t68 * (1.4799e-4  - t68 * (7.520e-6  - t68 * 5.5185e-8))
#    B = -1.1320e-5 - t68 * (8.7086-6  + t68 * (7.4936-7  - t68 * 2.6327e-8))
#    pv_sw = vpress_w(pt) * atm2mmhg / atm2pa + A * SP + (B * SP)**(3 / 2)
#    print(pv_sw)
#    print(A*SP)
#    print(B*SP**(3 / 2))
#    return vpress_w(pt) * atm2mmhg / atm2pa
#
#@match_args_return
#def vpress_w(t):
#    """
#    Water vapor pressure over fresh water
#    """
#    TK = t + K0
#
#    A = [-5800,1.391,-4.846e-2,4.176e-5,-1.445e-8,6.545]
#    pv_w = np.exp(A[0]/TK + A[1] + A[2]*TK + A[3]*TK**2 + A[4]*TK**3 + \
#                  A[5] * np.log(TK))
#    return pv_w

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
    # Xarray DataArrays do not support direct assignment (e.g. cd[u10 <= 11] = 0.0012)
    # ...but they do support np.where
    cd = np.where(u10 <= 11, 0.0012, cd)
    cd = np.where(u10 >= 20, 0.0018, cd)
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

@match_args_return
def kinematic_viscosity_air(air_temperature, air_density):
    """
    Calculate the kinematic viscosity of air given the air temperature and air density.
    Used in calculating air-side schmidt number (gasex.diff.air_side_Schmidt_number)

    Parameters:
        air_temperature (array-like): Air temperature in degrees Celsius.
        air_density (array-like): Air density in kg/m3

    Returns:
        float: Kinematic viscosity of air in m^2/s
    """
    # Convert air temperature to Kelvin
    T = air_temperature + K0
    
    rho_a = air_density # units are kg/m3
    
    # absolute viscosity of air (Sutherland's formula)
    mu_a = (1.458e-6 * T**1.5) / (T + 110.4) # units are kg/m/s
    
    # kinematic viscosity of air
    nu_a = mu_a / rho_a # units are m2/s
    
    return mu_a, nu_a

@match_args_return
def xH2O_from_rh(SP,pt,slp=1.0,rh=1.0):
    """
    Calculate the mixing ratio of H2O from relative humidity.
    Used in calculation of air density, which factors into air-side Schmidt number (gasex.airsea.L13)
    Parameters:
        SP (array-like): Practical salinity
        pt (array-like): potential temperature in degrees Celsius
        slp (array-like): sea level pressure in Atm (default: 1.0)
        rh (array-like): relative humidity (default: 1.0)

    Returns:
        mixing_ratio (array-like): mixing ratio of water in air in g/kg
    """    
    # Calculate vapor pressure at moist interface
    ph2oveq = vpress_sw(SP,pt) # atm
    ph2ov = rh * ph2oveq # atm
    
    molwt_h2o = 18.01528 # g/mol
    molwt_air = 28.96 # g/mol
    
    # Calculate mixing ratio
    mixing_ratio = molwt_h2o / molwt_air * ph2ov / (slp - ph2ov) * 1000  # Convert to g/kg

    return mixing_ratio

@match_args_return
def rh_from_dewpoint_SP_pt(SP, pt, dewpoint):
    """
    Calculate relative humidity from sea surface salinity, sea surface temperature,
    and air dewpoint temperature. Use to calculate relative humidity to feed into
    gas exchange calculation if you don't want to assume rh=100%.
    Parameters:
        SP (array-like): Practical salinity
        pt (array-like): potential temperature in degrees Celsius
        dewpoint (array-like): dewpoint temperature in degrees Celsius

    Returns:
        rh (array-like): relative humidity
    """  
    # Calculate vapor pressure at moist interface
    es = vpress_sw(SP,pt) # atm
    e = vpress_sw(SP,dewpoint) # atm
    
    rh = e/es
    
    return rh

@match_args_return
def rh_from_dewpoint_t2m(t2m, dewpoint):
    """
    Calculate relative humidity from air temperature and air dewpoint temperature.
    Use to calculate relative humidity to feed into gas exchange calculation
    if you don't want to assume rh=100%.
    Parameters:
        t2m (array-like): Air temperature at 2 meters in degrees Celsius
        dewpoint (array-like): dewpoint temperature at 2 meters in degrees Celsius

    Returns:
        rh (array-like): relative humidity
    """   
    # Calculate vapor pressure at moist interface
    es = vpress_w(t2m) # atm
    e = vpress_w(dewpoint) # atm
    
    rh = e/es
    
    return rh

@match_args_return
def calculate_air_density(temp_c, pressure_atm, mixing_ratio_g_kg):
    # Constants
    R_d = 287.058  # J/(kg·K) for dry air
    R_v = 461.495  # J/(kg·K) for water vapor
    epsilon = R_d / R_v

    # Convert input values to appropriate units
    T = temp_c + 273.15  # Convert temperature to Kelvin
    pressure_pa = pressure_atm * 101325  # Convert pressure to Pascals
    mixing_ratio = mixing_ratio_g_kg / 1000  # Convert mixing ratio to kg/kg

    # Calculate the partial pressures
    p_v = (mixing_ratio / (mixing_ratio + epsilon)) * pressure_pa
    p_d = pressure_pa - p_v

    # Calculate air density
    air_density = (p_d / (R_d * T)) + (p_v / (R_v * T))

    return air_density

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

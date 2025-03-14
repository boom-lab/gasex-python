#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:33:50 2017

Functions for calculating dissolved gas solubility

@author: d.nicholson
"""

from __future__ import division
import numpy as np
from gsw import pt_from_CT,SP_from_SA,CT_from_pt,rho
from ._utilities import match_args_return
from gasex.phys import K0 as K0
from gasex.phys import vpress_sw, R
from gasex.fugacity import fugacity_factor


__all__ = ['O2sol_SP_pt','Hesol_SP_pt','Nesol_SP_pt','Arsol_SP_pt', \
           'Krsol_SP_pt','N2sol_SP_pt','N2Osol_SP_pt','O2sol','Hesol', \
           'Nesol','Arsol','Krsol','N2sol','N2Osol']



@match_args_return
def eq_SP_pt(SP,pt,*,gas=None,slp=1.0,units="mM",chi_atm=None):
    """
    Description:
    -----------
    Wrapper function to calculate equilibrium solubility in [mol L-1] for gases
    with constant atmospheric mixing ratios.  Nobles gases, N2 and O2 are
    included.

    INPUT:
    -----------
    REQUIRED:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).
      gas =  String abbreviation for gas (He,Ne,Ar,Kr,Xe,O2 or N2)
    OPTIONAL:
      patm = total atmospheric pressure in atm (default = 1.0 atm)
      units = output units "M" or "umolkg" (default = M)

    OUTPUT:
    -----------
         soleq = gas eq. solubility including moist atmosphere             [M]

    AUTHOR: David Nicholson
    -----------
    """
    g_up = gas.upper()
    vp_sw = vpress_sw(SP,pt)
    SA = SP * 35.16504/35
    CT = CT_from_pt(SA,pt)
    dens = rho(SA,CT,0*CT)

    if chi_atm is None:
        chi_atm = air_mol_fract(gas=gas)
    
    if slp is None:
        p_corr = 1
    else:
        p_corr = (slp - vp_sw) / (1 - vp_sw)

    if gas == 'O2':
        soleq = O2sol_SP_pt(SP,pt)
    elif gas == 'He':
        soleq = Hesol_SP_pt(SP,pt)
    elif gas == 'Ne':
        soleq = Nesol_SP_pt(SP,pt)
    elif gas == 'Ar':
        soleq = Arsol_SP_pt(SP,pt)
    elif gas == 'Kr':
        soleq = Krsol_SP_pt(SP,pt)
    elif gas == 'Xe':
        soleq = Xesol_SP_pt(SP,pt)
    elif gas == 'N2':
        soleq = N2sol_SP_pt(SP,pt)
    elif gas == 'N2O':
        k = N2Osol_SP_pt(SP,pt)
        soleq = k*chi_atm*(1 - vp_sw) # mol/L
        soleq = soleq/dens*1000*1e6 # convert to umol/kg
    elif gas == 'CO2':
        k = CO2sol_SP_pt(SP,pt)
        soleq = k*chi_atm*(1 - vp_sw) # mol/L
        soleq = soleq/dens*1000*1e6 # convert to umol/kg
    elif gas == 'CH4':
        k = CH4sol_SP_pt(SP,pt)
        soleq = k*chi_atm*(1 - vp_sw) # mol/L
        soleq = soleq/dens*1000*1e6 # convert to umol/kg
    else:
        raise ValueError(gas + " is supported. Must be O2,He,Ne,Ar,Kr,Xe or \
                         N2")

    if units not in ("M","mM","uM","nM","molm3","umolkg"):
        raise ValueError("units: units must be \'M\','uM' or \'umolkg\'")

    if units == "M":
        eq =  p_corr * dens * soleq / 1e9
    elif units =="mM" or units =="molm3":
        eq =  p_corr * dens * soleq / 1e6
    elif units =="uM":
        eq =  p_corr * dens * soleq / 1e3
    elif units == "nM":
        eq =  p_corr * dens * soleq
    elif units == 'umolkg':
        eq =  p_corr * soleq
    return eq




@match_args_return
def sol_SP_pt(SP,pt,*,gas=None,p_dry=1.0,slp=1.0,rh=1.0,chi_atm=None,units="mM"):
    g_up = gas.upper()
    vp_sw = vpress_sw(SP,pt)*rh

    if g_up in ['O2','HE','NE','AR','KR','XE','N2']:
        if chi_atm is None:
            chi_atm = air_mol_fract(gas=gas)
        # solubility for 1 atm dry gas (K0)
        K0 = eq_SP_pt(SP,pt,gas=gas,units="M") / (chi_atm * (slp-vp_sw))
    elif gas == 'N2O':
        K0 = N2Osol_SP_pt(SP,pt)
    elif gas == 'CO2':
        K0 = CO2sol_SP_pt(SP,pt)
    elif gas == 'CH4':
        K0 = CH4sol_SP_pt(SP,pt)
    elif gas == 'CO':
        K0 = COsol_SP_pt(SP,pt)
    elif gas == 'H2':
        K0 = H2sol_SP_pt(SP,pt)
    else:
        raise ValueError(gas + " is not supported. Must be 'O2','He','Ne',\
                         'Ar','Kr','Xe','N2','CO2',N2O','CH4','CO' or 'H2'")
    if units == "M":
        return K0
    elif units =="mM" or units == "molm3":
        return K0 * 1e3
    elif units == 'umolkg':
        SA = SP * 35.16504/35
        CT = CT_from_pt(SA,pt)
        dens = rho(SA,CT,0*CT)
        return 1e-3 * K0 / dens
    else:
        raise ValueError("units: units must be \'M\', \'mM\',\'molm3\' or \'umolkg\'")


@match_args_return
def O2sol_SP_pt(SP,pt):

    """
     O2sol_SP_pt                              solubility of O2 in seawater
    ==========================================================================

     USAGE:
    import sol
    O2sol = sol.O2_SP_pt(35,20)

     DESCRIPTION:
      Calculates the oxygen concentration expected at equilibrium with air at
      an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
      saturated water vapor.  This function uses the solubility coefficients
      derived from the data of Benson and Krause (1984), as fitted by Garcia
      and Gordon (1992, 1993).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

     OUTPUT:
      O2sol = solubility of oxygen in micro-moles per kg           [ umol/kg ]

     AUTHOR:  David Nicholson
     Adapted from gsw_Arsol_SP_pt.m authored by Roberta Hamme, Paul Barker and
         Trevor McDougall

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Benson, B.B., and D. Krause, 1984: The concentration and isotopic
       fractionation of oxygen dissolved in freshwater and seawater in
       equilibrium with the atmosphere. Limnology and Oceanography, 29,
       620-632.

      Garcia, H.E., and L.I. Gordon, 1992: Oxygen solubility in seawater:
       Better fitting equations. Limnology and Oceanography, 37, 1307-1312.

      Garcia, H.E., and L.I. Gordon, 1993: Erratum: Oxygen solubility in
       seawater: better fitting equations. Limnology and Oceanography, 38,
       656.

      The software is available from http://www.TEOS-10.org

    ==========================================================================
    """


    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024     # pt68 is the potential temperature in degress C on
              # the 1968 International Practical Temperature Scale IPTS-68.
    y = np.log((25+K0 - pt68)/(K0 + pt68))

# The coefficents below are from the second column of Table 1 of Garcia and
# Gordon (1992) for fit to Benson and Krause (1984)
    a = (5.80871,3.20291,4.17887,5.10006,-9.86643e-2,3.80369)
    b = (-7.01577e-3,-7.70028e-3,-1.13864e-2,-9.51519e-3)
    c = -2.75915e-7


    lnC = (a[0] + y * (a[1] + y * (a[2] + y * (a[3] + y * (a[4] + a[5] * y))))\
          + x * (b[0] + y * (b[1] + y * (b[2] + b[3] * y)) + c * x))

    return np.exp(lnC)


@match_args_return
def O2sol(SA,CT,p,long,lat):
    """
     O2      Solubility of O2 in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return O2sol_SP_pt(SP,pt)


@match_args_return
def Hesol_SP_pt(SP,pt):
    """
     Hesol_SP_pt                              solubility of He in seawater
    ==========================================================================

     USAGE:
      Hesol = Hesol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the helium concentration expected at equilibrium with air at
      an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
      saturated water vapor.  This function uses the solubility coefficients
      as listed in Weiss (1971).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:

      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      Hesol = solubility of helium in micro-moles per kg           [ umol/kg ]

     AUTHOR:  David Nicholson
     Adapted from gsw_Arsol_SP_pt.m authored by Roberta Hamme, Paul Barker and
         Trevor McDougall

     VERSION NUMBER: 3.05 (27th January 2015)

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Dymond and Smith, 1980: The virial coefficients of pure gases and
       mixtures. Clarendon Press, Oxford.

      Weiss, R.F., 1971: Solubility of Helium and Neon in Water and Seawater.
       J. Chem. and Engineer. Data, 16, 235-241.


    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # The coefficents below are from Table 3 of Weiss (1971)
    a = (-167.2178, 216.3442, 139.2032, -22.6202)
    b = (-0.044781, 0.023541, -0.0034266)

    Hesol_mL = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) + a[3] * \
                      y_100 + x * (b[0] + y_100 * (b[1] + b[2] * y_100)))

    Hesol = 1000* Hesol_mL / mol_vol(gas="He")
    # mL/kg to umol/kg for He (1/22.44257e-3)
    #Molar volume at STP (Dymond and Smith, 1980).
    return Hesol


@match_args_return
def Hesol(SA,CT,p,long,lat):
    """
     He      Solubility of He in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return Hesol_SP_pt(SP,pt)

@match_args_return
def Nesol_SP_pt(SP,pt):
    """
     Nesol_SP_pt                              solubility of Ne in seawater
    ==========================================================================

     USAGE:
      Nesol = gsw_Nesol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the Neon, Ne, concentration expected at equilibrium with air
      at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
      saturated water vapor.  This function uses the solubility coefficients
      as listed in Hamme and Emerson (2004).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      Nesol = solubility of neon in umol per kg              [ umol/kg ]

     AUTHOR:  David Nicholson
     Adapted from gsw_Arsol_SP_pt.m authored by Roberta Hamme, Paul Barker and
         Trevor McDougall

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Hamme, R., and S. Emerson, 2004: The solubility of neon, nitrogen and
       argon in distilled water and seawater. Deep-Sea Research, 51,
       1517-1528.



    ==========================================================================
    """
    x = SP
    # Note that salinity argument is Practical Salinity, this is
    # beacuse the major ionic components of seawater related to Cl
    # are what affect the solubility of non-electrolytes in seawater.

    y = np.log((25+K0 - pt)/(K0 + pt))
    # pt is the temperature in degress C on the ITS-90 scale

    # The coefficents below are from Table 4 of Hamme and Emerson (2004)
    a =  (2.18156, 1.29108, 2.12504)
    b = (-5.94737e-3, -5.13896e-3)

    # umol kg-1 for consistency with other gases
    Nesol = np.exp(a[0] + y * (a[1] + a[2] * y) + x * (b[0] + b[1] * y)) / 1e3
    return Nesol

@match_args_return
def Nesol(SA,CT,p,long,lat):
    """
     Ne      Solubility of Ne in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return Nesol_SP_pt(SP,pt)

@match_args_return
def Arsol_SP_pt(SP,pt):
    """
     Arsol_SP_pt                              solubility of Ar in seawater
    ==========================================================================

     USAGE:
      Arsol = gsw_Arsol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the argon, Ar, concentration expected at equilibrium with air
      at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar) including
      saturated water vapor  This function uses the solubility coefficients
      as listed in Hamme and Emerson (2004).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      Arsol = solubility of argon                                  [ umol/kg ]

     AUTHOR:  David Nicholson
     Adapted from gsw_Arsol_SP_pt.m authored by Roberta Hamme, Paul Barker and
         Trevor McDougall

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Hamme, R., and S. Emerson, 2004: The solubility of neon, nitrogen and
       argon in distilled water and seawater. Deep-Sea Research, 51,
       1517-1528.


    ==========================================================================
    """

    x = SP
    # Note that salinity argument is Practical Salinity, this is
    # beacuse the major ionic components of seawater related to Cl
    # are what affect the solubility of non-electrolytes in seawater.
    #pt68 = pt * 1.00024     # pt68 is the potential temperature in degress C on
              # the 1968 International Practical Temperature Scale IPTS-68.
    y = np.log((25+K0 - pt)/(K0 + pt))
    # pt is the temperature in degress C on the ITS-90 scale

    # The coefficents below are from Table 4 of Hamme and Emerson (2004)
    a =  (2.79150, 3.17609, 4.13116, 4.90379)
    b = (-6.96233e-3, -7.66670e-3, -1.16888e-2)

    Arsol = np.exp(a[0] + y * (a[1] + y * (a[2] + a[3] * y)) + x * \
                   (b[0] + y *(b[1] + b[2] *y )))
    return Arsol

@match_args_return
def Arsol(SA,CT,p,long,lat):
    """
     Ar      Solubility of Ar in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return Arsol_SP_pt(SP,pt)

@match_args_return
def Krsol_SP_pt(SP,pt):
    """
     Krsol_SP_pt                              solubility of Kr in seawater
    ==========================================================================

     USAGE:
      Krsol = sol.Krsol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the krypton, Kr, concentration expected at equilibrium with
      air at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar)
      including saturated water vapor.  This function uses the solubility
      coefficients derived from the data of Weiss (1971).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      Krsol = solubility of krypton in micro-moles per kg          [ umol/kg ]

     AUTHOR:  Roberta Hamme, Paul Barker and Trevor McDougall
                                                          [ help@teos-10.org ]


     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Weiss, R.F. and T.K. Kyser, 1978: Solubility of Krypton in Water and
       Seawater. J. Chem. Thermodynamics, 23, 69-72.

      The software is available from http://www.TEOS-10.org

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 2 (Weiss and Kyser, 1978)
    a = (-112.6840, 153.5817, 74.4690, -10.0189)
    b = (-0.011213, -0.001844, 0.0011201)

    Krsol_mL = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) + a[3] * \
                      y_100 + x * (b[0] + y_100 * (b[1] + b[2] * y_100)))

    # mL/kg to umol/kg for Kr (1/22.3511e-3)
    #Molar volume at STP (Dymond and Smith, 1980).
    Krsol = Krsol_mL * 4.474052731185490e1
    return Krsol

@match_args_return
def Krsol(SA,CT,p,long,lat):
    """
     Kr      Solubility of Kr in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return Krsol_SP_pt(SP,pt)


@match_args_return
def Xesol_SP_pt(SP,pt):
    """
     Xesol_SP_pt                              solubility of Xe in seawater
    ==========================================================================

     USAGE:
      Xesol = sol.Xesol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the xenon (Xe) concentration expected at equilibrium with
      air at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar)
      including saturated water vapor.


     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      Xesol = solubility of xenon at 1 atm moist atmosphere        [ umol/kg ]

     AUTHOR:  D. Nicholson  Adapted from MATLAB Xesol.m by R. Hamme


     REFERENCES:
         R. Hamme fit to data of D. Wood and R. Caputi (1966) "Solubilities of
         Kr and Xe in fresh and sea water" U.S. Naval Radiological Defense
         Laboratory, Technical Report USNRDL-TR-988,San Francisco, CA, pp. 14.

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.
    pt68 = pt * 1.00024     # pt68 is the potential temperature in degress C on
              # the 1968 International Practical Temperature Scale IPTS-68.
    y = np.log((25+K0 - pt68)/(K0 + pt68))
    # pt is the temperature in degress C on the ITS-90 scale


    #  from fit procedure of Hamme and Emerson 2004 to Wood and Caputi data
    a = (-7.48588, 5.08763, 4.22078)
    b = (-8.17791e-3, -1.20172e-2)

    Xesol = np.exp(a[0] + y * (a[1] + y * a[2]) + x * (b[0] + y *b[1]))

    return Xesol

@match_args_return
def Xesol(SA,CT,p,long,lat):
    """
     Xe      Solubility of Xe in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return Xesol_SP_pt(SP,pt)


@match_args_return
def N2sol_SP_pt(SP,pt):
    """
     N2sol_SP_pt                              solubility of N2 in seawater
    ==========================================================================

     USAGE:
      N2sol = sol.N2sol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the nitrogen, N2, concentration expected at equilibrium with
      air at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar)
      including saturated water vapor.  This function uses the solubility
      coefficients as listed in Hamme and Emerson (2004).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      N2sol = solubility of nitrogen in micro-moles per kg         [ umol/kg ]

     AUTHOR:  Roberta Hamme, Paul Barker and Trevor McDougall
                                                          [ help@teos-10.org ]

     VERSION NUMBER: 3.05 (27th January 2015)

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Hamme, R., and S. Emerson, 2004: The solubility of neon, nitrogen and
       argon in distilled water and seawater. Deep-Sea Research, 51,
       1517-1528.

      The software is available from http://www.TEOS-10.org

    ==========================================================================
"""
    x = SP
    # Note that salinity argument is Practical Salinity, this is
    # beacuse the major ionic components of seawater related to Cl
    # are what affect the solubility of non-electrolytes in seawater.

    y = np.log((25+K0 - pt)/(K0 + pt))
    # pt is the temperature in degress C on the ITS-90 scale

    # The coefficents below are from Table 4 of Hamme and Emerson (2004)
    a = (6.42931, 2.92704, 4.32531, 4.69149)
    b = (-7.44129e-3, -8.02566e-3, -1.46775e-2)

    N2sol = np.exp(a[0] + y * (a[1] + y * (a[2] + a[3] * y)) + x * \
                   (b[0] + y *(b[1] + b[2] *y )))
    return N2sol

@match_args_return
def N2sol(SA,CT,p,long,lat):
    """
     N2      Solubility of N2 in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return N2sol_SP_pt(SP,pt)

@match_args_return
def N2Osol_SP_pt(SP,pt):
    """
     gsw_N2Osol_SP_pt                            solubility of N2O in seawater
    ==========================================================================

     USAGE:
      N2Osol = gsw_N2Osol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the nitrous oxide, N2O, concentration expected at equilibrium
      with air at an Absolute Pressure of 101325 Pa (sea pressure of 0 dbar)
      including saturated water vapor  This function uses the solubility
      coefficients as listed in Hamme and Emerson (2004).

      Note that this algorithm has not been approved by IOC and is not work
      from SCOR/IAPSO Working Group 127. It is included in the GSW
      Oceanographic Toolbox as it seems to be oceanographic best practice.

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      N2Osol = K' solubility of nitrous oxide                 mol L-1 atm-1 ]
              (solubility in moist air at total pressure of 1 atm)
              coefficients are also included for output in mol kg-1 atm-1

     AUTHOR:  Rich Pawlowicz, Paul Barker and Trevor McDougall
                                                          [ help@teos-10.org ]

     VERSION NUMBER: 3.05 (27th January 2015)

     REFERENCES:
      IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
       seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

      Weiss, R.F. and B.A. Price, 1980: Nitrous oxide solubility in water and
       seawater. Mar. Chem., 8, 347-359.
       https://doi.org/10.1016/0304-4203(80)90024-9

      The software is available from http://www.TEOS-10.org

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # The coefficents below are from Table 2 of Weiss and Price (1980)
    
    # These coefficients are for mol L-1 atm-1
    a = (-62.7062, 97.3066, 24.1406)
    b = (-0.058420, 0.033193, -0.0051313)

    # These coefficients are for mol kg-1 atm-1
    # a = (-64.8539, 100.2520, 25.2049)
    # b = (-0.062544, 0.035337, -0.0054699)

    # Moist air correction at 1 atm.
    # fitted the vapor pressure of water as given by Goff and Gratch (1946),
    # and the vapor pressure lowering by sea salt as given by Robinson (1954),
    # to a polynomial in temperature and salinity:

    #m = [24.4543, 67.4509, 4.8489, 0.000544]
    #ph2odP = np.exp(m[0] - m[1]*100/y - m[2] * np.log(y_100) - m[3] * x)

    N2Osol = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) + x * \
                     (b[0] + y_100 * (b[1] + b[2] * y_100)))
    return N2Osol

@match_args_return
def N2Osol(SA,CT,p,long,lat):
    """
     N2O     Solubility of N2O in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return N2Osol_SP_pt(SP,pt)

@match_args_return
def CO2sol_SP_pt(SP,pt):
    """
     CO2sol_SP_pt            solubility of CO2 in seawater for 1 atm moist air
    ==========================================================================

     USAGE:
      import gas.sol as sol
      CO2sol = sol.CO2sol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates the carbon dioxide, CO2, concentration expected at equilibrium
      with a pure CO2 pressure of 101325 Pa (1.0 atm) This function uses the
      solubility coefficients derived from the data of Weiss (1974)


     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      CO2sol = solubility of carbon dioxide at 1 atm dry     [ mol L-1 atm-1 ]

     AUTHOR:  David Nicholson
                                                        [ dnicholson@whoi.edu ]


     REFERENCES:
      Weiss, R.F. and B.A. Price, 1980: Nitrous oxide solubility in water and
        seawater. Mar. Chem., 8, 347-359.
        https://doi.org/10.1016/0304-4203(80)90024-9
      Weiss, R. (1974) Carbon Dioxide in Water and Seawater The Solubility of a
        Non- Ideal Gas. Mar. Chem., 2, 203-215.
        https://doi.org/10.1016/0304-4203(74)90015-2

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 6 (Weiss and Price, 1980)
    #a = [-162.8301, 218.2968, 90.9241, -1.47696]
    #b = [0.025695, -0.025225, 0.0049867]
    # Table 1 (Weiss 1974, Marine Chem)
    a = (-58.0931, 90.5069, 22.2940)
    b = (0.027766, -0.025888, 0.0050578)

    CO2sol = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) +  x * \
                    (b[0] + b[1] * y_100 + b[2] * y_100**2))
    return CO2sol

@match_args_return
def CO2sol(SA,CT,p,long,lat):
    """
     CO2    Solubility of CO2 in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return CO2sol_SP_pt(SP,pt)

@match_args_return
def CH4sol_SP_pt(SP,pt):
    """
     CH4sol_SP_pt            solubility of CH4 in seawater  [mol L-1 atm-1]
    ==========================================================================

     USAGE:
      import gas.sol as sol
      CH4sol = sol.CH4sol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates methane solubility (K0) in units of mol L-1 atm-1. Equivalent
      to the dissloved concentration in equilibrium with a pure CH4 pressure
      of 101325 Pa (1.0 atm) This function uses the solubility coefficients
      from Weisenburg and Guinasso fit to data from Yamamoto et al. 1976.


     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      CH4sol = solubility of methane                          [ mol L-1 atm-1 ]

     AUTHOR:  David Nicholson
                                                        [ dnicholson@whoi.edu ]


     REFERENCES:
      Wiesenburg, D. A., and N. L. Guinasso (1979), Equilibrium solubilities of
          methane, carbon monoxide, and hydrogen in water and sea water,
          J.Chem. Eng. Data, 24(4), 356?360, doi:10.1021/je60083a006.

      Yamamoto, S., J. B. Alcauskas, and T. E. Crozier (1976), Solubility of
          methane in distilled water and seawater, J. Chem. Eng. Data, 21(1),
          78?80, doi:10.1021/je60068a029.

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 1 in Weisenburg and Guinasso 1979
    a = (-68.8862, 101.4956, 28.7314)
    b = (-0.076146, 0.043970, -0.0068672)

    # Bunsen solubility in cc gas @STP / mL H2O atm-1
    CH4_beta = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) +  x * \
                    (b[0] + b[1] * y_100 + b[2] * y_100**2))
    # Divide by gas virial volume to get mol L-1 atm-1
    CH4sol = CH4_beta / mol_vol(gas='CH4')
    return CH4sol

@match_args_return
def CH4sol(SA,CT,p,long,lat):
    """
     CH4    Solubility of CH4 in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return CH4sol_SP_pt(SP,pt)

@match_args_return
def COsol_SP_pt(SP,pt):
    """
     COsol_SP_pt            solubility of CO in seawater  [mol L-1 atm-1]
    ==========================================================================

     USAGE:
      import gas.sol as sol
      COsol = sol.COsol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates carbon monoxide solubility (K0) in units of mol L-1 atm-1.
      Equivalent to the dissloved concentration in equilibrium with a pure CO
      pressure of 101325 Pa (1.0 atm) This function uses the solubility
      coefficients from Weisenburg and Guinasso fit to data from Douglas (1967)
      and Winkler (1906).

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      COsol = solubility of carbon monoxide                   [ mol L-1 atm-1 ]

     AUTHOR:  David Nicholson
                                                        [ dnicholson@whoi.edu ]


     REFERENCES:
      Wiesenburg, D. A., and N. L. Guinasso (1979), Equilibrium solubilities of
          methane, carbon monoxide, and hydrogen in water and sea water,
          J.Chem. Eng. Data, 24(4), 356?360, doi:10.1021/je60083a006.

      Douglas, E., J. Phys. Chem., 71, 1931 (1967).

      Winkler, I.W., 2.Phys. Chem. Abt. A, 55, 344 (1906)

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 1 in Weisenburg and Guinasso 1979
    a = (-47.6148, 69.5068, 18.7397)
    b = (0.045657, -0.040721, 0.0079700)

    # Bunsen solubility in cc gas @STP / mL H2O atm-1
    CO_beta = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) +  x * \
                    (b[0] + b[1] * y_100 + b[2] * y_100**2))
    # Divide by gas virial volume to get mol L-1 atm-1
    COsol = CO_beta / mol_vol(gas='CO')
    return COsol

@match_args_return
def COsol(SA,CT,p,long,lat):
    """
     CO    Solubility of CO in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return COsol_SP_pt(SP,pt)

@match_args_return
def H2sol_SP_pt(SP,pt):
    """
     H2sol_SP_pt            solubility of H2 in seawater  [mol L-1 atm-1]
    ==========================================================================

     USAGE:
      import gas.sol as sol
      COsol = sol.COsol_SP_pt(SP,pt)

     DESCRIPTION:
      Calculates hydrogen gas solubility (K0) in units of mol L-1 atm-1.
      Equivalent to the dissloved concentration in equilibrium with a pure H2
      pressure of 101325 Pa (1.0 atm) This function uses the solubility
      coefficients from Weisenburg and Guinasso fit to data from Douglas (1967)
      and Winkler (1906).

     INPUT:
      SP  =  Practical Salinity  (PSS-78)                         [ unitless ]
      pt  =  potential temperature (ITS-90) referenced               [ deg C ]
             to one standard atmosphere (0 dbar).

      SP & pt need to have the same dimensions.

     OUTPUT:
      H2sol = solubility of hydrogen gas                      [ mol L-1 atm-1 ]

     AUTHOR:  David Nicholson
                                                        [ dnicholson@whoi.edu ]


     REFERENCES:
      Wiesenburg, D. A., and N. L. Guinasso (1979), Equilibrium solubilities of
          methane, carbon monoxide, and hydrogen in water and sea water,
          J.Chem. Eng. Data, 24(4), 356?360, doi:10.1021/je60083a006.

      Crozier, T. E., Yamamoto, S.,J. Chem. Eng. Data, 19, 242 (1974)

      Gordon, L. I., Cohen, Y., Standley, D. R., Deep-sea Res., 24, 937 (1977)

    ==========================================================================
    """
    x = SP        # Note that salinity argument is Practical Salinity, this is
             # beacuse the major ionic components of seawater related to Cl
          # are what affect the solubility of non-electrolytes in seawater.

    pt68 = pt * 1.00024 # pt68 is the potential temperature in degress C on
                  # the 1968 International Practical Temperature Scale IPTS-68.
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 1 in Weisenburg and Guinasso 1979
    a = (-47.8948, 65.0368, 20.1709)
    b = (-0.082225, 0.049564, -0.0078689)

    # Bunsen solubility in cc gas @STP / mL H2O atm-1
    H2_beta = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) +  x * \
                    (b[0] + b[1] * y_100 + b[2] * y_100**2))
    # Divide by gas virial volume to get mol L-1 atm-1
    H2sol = H2_beta / mol_vol(gas='H2')
    return H2sol

@match_args_return
def H2sol(SA,CT,p,long,lat):
    """
     H2    Solubility of H2 in seawater from absolute salinity and cons temp
    ==========================================================================
    """
    SP = SP_from_SA(SA,p,long,lat)
    pt = pt_from_CT(SA,CT)
    return H2sol_SP_pt(SP,pt)



def air_mol_fract(gas=None):
    """molar mixing ratio for well-mixed gases

    Args:
        gas ([string], optional): abbreviation for gas. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        xG: molar atmospheric mixing ratio for well-lixed gases ('O2','HE','NE','AR','KR','XE' or 'N2')
    """
    g_up = gas.upper()
    if g_up in ['O2','HE','NE','AR','KR','XE','N2']:
        frac_dict = {'O2':np.array([0.209790]), \
                 'HE':np.array([5.24e-6]), \
                 'NE':np.array([0.00001818]), \
                 'AR':np.array([0.009332]), \
                 'KR':np.array([0.00000114]), \
                 'XE':np.array([8.7e-8]), \
                 'N2':np.array([0.780848]) }
        return frac_dict[g_up]
    else:
        raise ValueError(f"must specify chi_atm for {g_up}. Default chi_atm only available for O2, He, Ne, Ar, Kr, Xe, and N2.")  

def mol_vol(gas=None):
    g_up = gas.upper()
    vol_dict = {'HE':np.array([22.4263]), \
                 'NE':np.array([22.4241]), \
                 'AR':np.array([22.3924]), \
                 'KR':np.array([22.3518]), \
                 'XE':np.array([22.2582]), \
                 'O2':np.array([22.3922]), \
                 'N2':np.array([22.4045]), \
                 'N2O':np.array([22.243]), \
                 'CO2':np.array([0.99498*22.414]), \
                 'CH4':np.array([22.360]), \
                 'H2':np.array([22.428]) }
    return vol_dict[g_up]

#def mol_vol_calc(gas=None,t=298.15):
#    """
#    http://www.kayelaby.npl.co.uk/chemistry/3_5/3_5.html
#    """
#    B_dict = {'CO': [202.6,154.2,94.2]}
#    bcoeff = B_dict[gas]
#    B = bcoeff[0] - bcoeff[1] * np.exp(bcoeff[2]*t / 298.15)
#
#    # quadratic coefficients
#    a = 1 / (R * 298.15)
#    b = -1
#    c = -B
#    print (B)
#    print(a)
#    V = b + np.sqrt(1 - 4 * a * c) / 2 * a
#    return V

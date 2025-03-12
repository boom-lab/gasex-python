#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:03:01 2017

@author: dnicholson
"""

from gasex import sol,diff
import unittest

class TestCheckVals(unittest.TestCase):
    """
    Test solubility check value
    """

    def test_O2eq(self):
        """
        Test O2 sol check value from MATLAB gsw_O2sol_SP_pt.m
        differs slightly from Garcia Gordon (1992) 274.10 check value
        this is b/c of pt68 --> pt temperature conversion
        """
        tolx = 1e-8
        result = sol.eq_SP_pt(35,10,gas='O2',units='umolkg')
        self.assertTrue(abs(result/274.5956644859288 - 1) < tolx)
        
    
    def test_Neeq(self):
        """
        Check value from Hamme and Emerson 2004
        """
        tolx = 1e-5
        result = sol.eq_SP_pt(35,10,gas='Ne',units='umolkg')
        self.assertTrue(abs(result/0.00734121 - 1) < tolx)
        
    def test_Areq(self):
        """
        Check value from Hamme and Emerson 2004
        """
        tolx = 1e-5
        result = sol.eq_SP_pt(35,10,gas='Ar',units='umolkg')
        self.assertTrue(abs(result/13.4622 - 1) < tolx)
    
    def test_Kreq(self):
        """
        Check value against gsw_Krsol
        """
        tolx = 1e-8
        result = sol.eq_SP_pt(35,10,gas='Kr',units='umolkg')
        self.assertTrue(abs(result/0.003137398904939 - 1) < tolx)
        
    def test_N2eq(self):
        """
        Check value from Hamme and Emerson 2004
        """
        tolx = 1e-5
        result = sol.eq_SP_pt(35,10,gas='N2',units='umolkg')
        self.assertTrue(abs(result/500.885 - 1) < tolx)
        
    def test_CO2beta(self):
        """
        Check values from Weiss 1974 Table II
        """
        S = (0, 10, 20, 30, 35)
        n = len(S)
        checkvals = (5.366e-2, 5.105e-2, 4.857e-2, 4.621e-2, 4.507e-2) 
        tolx = 1e-4
        result = sol.CO2sol_SP_pt(S,10 / 1.00024)
        for i in range(n):
            self.assertTrue(abs(result[i]/checkvals[i] - 1) < tolx)
    
    def test_diff(self):
        """
        Test gas diffusion for S=35, T=20
        """
        tolx = 1e-8
        gases = ('He','Ne','Ar','Kr','Xe','N2','O2','CH4','CO2')
        d_check = (6.4052495707474602e-09,3.4720163930751105e-09, \
                   2.2602874175879768e-09,1.530920024417149e-09, \
                   1.2094650981458935e-09,1.6412087070348969e-09,\
                   1.8992006052350516e-09,1.552288805924663e-09,\
                   1.5951796366083652e-09)
        ng = len(gases)
        for i in range(ng):
            result = diff.diff(35,20,gas=gases[i])
            self.assertTrue(abs(result/d_check[i] - 1) < tolx)
    
    def test_schmidt(self):
        """
        Test schmidt number check values
        """
        gases = ('CO2','N2O','CH4','RN','SF6','DMS','CFC12','CFC11','CH3BR','CCL4')
        sc_check_sw = {'CO2': 668,
            'N2O': 697,
            'CH4':687,
            'RN': 985,
            'SF6':1028,
            'DMS':941,
            'CFC12':1188,
            'CFC11':1179,
            'CH3BR':701,
            'CCL4': 1315 }
        sc_check_fw = {'CO2': 600,
            'N2O': 626,
            'CH4':617,
            'RN': 884,
            'SF6':953,
            'DMS':844,
            'CFC12':1066,
            'CFC11':1126,
            'CH3BR':670,
            'CCL4': 1181 }
        for gas in sc_check_sw.keys():
            sw_result = diff.schmidt_W14(20,gas=gas,sw=True)
            fw_result = diff.schmidt_W14(20,gas=gas,sw=False)
            self.assertTrue(round(sw_result)==sc_check_sw[gas])
            self.assertTrue(round(fw_result)==sc_check_fw[gas])

if __name__ == '__main__':
    unittest.main()

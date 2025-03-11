from gasex.airsea import L13
from gasex.sol import sol_SP_pt,eq_SP_pt, air_mol_fract
from gasex.phys import vpress_sw

(Fd, Fc, Fp, Deq, Ks) = L13(0.01410,5,35,10,slp=1.0,gas='Ar',rh=0.9,
				calculate_schmidtair=False)#, chi_atm=0.009332)
print(Fd)
print(Fc)
print(Fp)
print(Deq)
print(Ks)

# expected values
expected_Fd = -5.2632e-09 #-5.2641e-09
expected_Fc = 1.3605e-10 #1.3605e-10
expected_Fp = -6.0079e-10 #-6.0093e-10
expected_Deq = 0.0014 #0.0014
expected_Ks = 2.0374e-05 #2.0377e-05

        # assertions with tolerance
print(abs(Fd / expected_Fd - 1))
print(abs(Fc / expected_Fc - 1))
print(abs(Fp / expected_Fp - 1))
print(abs(Deq / expected_Deq - 1))
print(abs(Ks / expected_Ks - 1))

xG = air_mol_fract(gas='Ar')
Geq1 = eq_SP_pt(35,5,gas='Ar',units="mM")
ph2oveq = vpress_sw(35,5) # atm
Geq2 = 0.009332 * sol_SP_pt(35,5,gas='Ar',units="mM") * (1 - ph2oveq)

print(Geq1 - Geq2)
"""
s = sol_SP_pt(35,10,chi_atm=3.37e-7, gas="N2O",units="mM")
C = 13e-6
pN2Osw = C/s

(Fd, Fc, Fp, Deq, Ks) = L13(C,5,35,10,slp=1.0,gas='N2O',rh=1.0, chi_atm=3.37e-7)
#(Fd, Fc, Fp, Deq, Ks) = L13(pN2Osw,5,35,10,slp=1.0,gas='N2O',rh=1.0, chi_atm=3.37e-7,
	#pressure_mode = True)
print(Fd*1e6*86400) # convert fluxes from mol m-2 s-1 to umol/m2/day (convention for N2O)
print(Fc*1e6*86400)
print(Fp*1e6*86400)
print(Deq)
print(Ks)
"""
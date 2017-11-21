# gasex-python
**python tools for dissolved gases and air-sea exchange in oceanography**

Tools designed for dissolved gases for oceanographic applications for a suite of insoluble gases
Gases included so far are: O2, CO2, CH4, CO, N2, N2O, He, Ne, Ar, Kr, Xe.  Not all functions are implemented yet for all gases

gasex-python is designed to integrate with the python implemntation of the TEOS-10 gsw-toolbox (https://github.com/TEOS-10/GSW-Python)
functions are available for either absolute  salinity (SA) and conservative temperature (CT), or for practical salinity (SP) scale and potential temperature (pt)

## Installation

gasex-python is under active development and there is not an official release available yet.  To work with the latest version you should be able to pip install . after cloning the repository:

>>> git clone https://github.com/dnicholson/gasex-python .
>>> pip install .

## Contents

**sol.py**:         solubility functions and atmospheric composition of gases
**wrapper functions**
    **sol_SP_pt**:  calculates gas solubility (K0) for gases. Default units mol L<sup>-1</sup> atm<sup>-1</sup>

    **eq_SP_pt**:  calculates gas conc. in equilibrium with moist atmosphere at 1 atm total pressure applicable only for gases with constant atmospheric mixing ratios (noble gases, N2, O2). default units mol L<sup>-1</sup>.

**gas specific functions**
    for each gas, a function of the form:

    **O2sol_SP_pt**: specific solubility functions for each gas. Output can be equilibrium solubility (F), dry solubility (K0) or moist air solubility (K'). The goal is to provide output in the most commonly used format for each gas as it appears in the literature, and to reproduce original publications.  For example, for O<sub>-2</sub>, default output is umol kg<sup>-1</sup>.  For CO2, output is K0 in mol L<sup>-1</sup> atm<sup>-1</sup>

    **O2sol**: Same as above, but for SA and CT

**diff.py**:    Diffusion and Schmidt number of gases

    **diff**:           Diffusivity of gases in seawater as a function of temperature and salinity

    **schmidt**:    Schmidt number of gases as a function of temperature and salinity

**fas.py**:         Air-sea flux based on a range of published wind-speed based parameterizations

    **fas**:          Air-sea surface diffusive flux based on dissolved concentration (for gases with constant atm mixing ratio)

    **fas_pC**:    Air-sea surface diffusive flux based on partial pressure gradient
    TODO:           Parameterizations including bubble-mediated gas transfer

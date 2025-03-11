import unittest
from gasex import airsea

class TestL13Function(unittest.TestCase):
    """
    Unit tests for the L13 function in the gasex module.
    """

    def test_L13_Ar(self):
        """
        Test L13 function with gas='Ar'. 
        This test checks the expected values of Fd, Fc, Fp, Deq, and Ks 
        when the gas is Argon (Ar), with given input parameters.
        """
        tolx = 1e-10
        (Fd, Fc, Fp, Deq, Ks) = airsea.L13(0.01410, 5, 35, 10, slp=1.0, gas='Ar', rh=0.9)

        # Expected values
        expected_Fd = -5.2641e-09
        expected_Fc = 1.3605e-10
        expected_Fp = -6.0093e-10
        expected_Deq = 0.0014
        expected_Ks = 2.0377e-05

        # Assertions with tolerance
        self.assertTrue(abs(Fd / expected_Fd - 1) < tolx)
        self.assertTrue(abs(Fc / expected_Fc - 1) < tolx)
        self.assertTrue(abs(Fp / expected_Fp - 1) < tolx)
        self.assertTrue(abs(Deq / expected_Deq - 1) < tolx)
        self.assertTrue(abs(Ks / expected_Ks - 1) < tolx)

if __name__ == "__main__":
    unittest.main()

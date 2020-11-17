#calculate the parameters in ADK model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import gamma
from scipy.constants import c, e, m_e, physical_constants
from fbpic.particles.elementary_process.ionization.read_atomic_data import get_ionization_energies

class ADK_calculator(object):
    def __init__( self, element, level_max=None ):
        """
        This is mostly copied from fbpic/particles/elementary_process/ionization/ionizer.py
        but with some modifications.
        Initialize parameters needed for the calculation of ADK ionization rate

        Parameters
        ----------
        element: string
            The atomic symbol of the considered ionizable species
            (e.g. 'He', 'N' ;  do not use 'Helium' or 'Nitrogen')

        See Chen, JCP 236 (2013), equation (2) for the ionization rate formula
        """
        self.set_element(element, level_max)

    def set_element( self, element, level_max=None ):
        # Get the array of energies
        Uion = get_ionization_energies( element )
        # Check whether the element string was valid
        if Uion is None:
            raise ValueError("Unknown ionizable element %s.\n" %element + \
            "Please use atomic symbol (e.g. 'He') not full name (e.g. Helium)")
        else:
            self.element = element

        # Determine and set the maximum level of ionization
        self.level_max = level_max
        if self.level_max is None:
            self.level_max = len(Uion)
        else:
            assert type(self.level_max) is int, "level_max must be integer"
            if self.level_max>len(Uion):
                raise ValueError("Chosen level_max for {}".format(element) + \
                                 " cannot exceed {}".format(len(Uion)))

        # Calculate the ADK prefactors (See Chen, JCP 236 (2013), equation (2))
        # - Scalars
        alpha = physical_constants['fine-structure constant'][0]
        r_e = physical_constants['classical electron radius'][0]
        wa = alpha**3 * c / r_e
        Ea = m_e*c**2/e * alpha**4/r_e
        # - Arrays (one element per ionization level)
        UH = get_ionization_energies('H')[0]
        Z = np.arange( len(Uion) ) + 1
        self.n_eff = n_eff = Z * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1
        self.C2 = C2 = 2**(2*n_eff) / (n_eff * gamma(n_eff+l_eff+1) * gamma(n_eff-l_eff))
        # For now, we assume l=0, m=0
        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = wa * C2 * ( Uion/(2*UH) ) \
            * ( 2*(Uion/UH)**(3./2)*Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea

    def get_ADK_parameters(self):
        return np.transpose([self.adk_prefactor, self.adk_exp_prefactor, self.adk_power])

    def ionization_rate(self, E, level=None):
        '''
            return the ionization rate
        '''
        if level is None:
            return self.adk_prefactor * np.exp(self.adk_exp_prefactor/E) * E**self.adk_power
        else:
            return self.adk_prefactor[level] * np.exp(self.adk_exp_prefactor[level]/E) * E**self.adk_power[level]

if __name__ == '__main__':
    ion=ADK_calculator('O')
    print(ion.get_ADK_parameters())

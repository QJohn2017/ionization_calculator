#calculate ioniztion probability for variety of elements and ionization states
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.constants import c
import ADK_calculator
# Do not have to start X-server. Save figures instead of show them.
import matplotlib as mpl
mpl.use('Agg')

osiris_ionization_parameters = {\
            'H':[[8.522542995398661e19, 342.53947239007687e0, 1.0005337056631487e0]],\
            'He':[[7.2207661763501e18, 832.809878216992e0, 0.48776427204592254e0],\
                  [2.7226733893691e21, 2742.1316798375965e0, 1.0000920088118899e0]],\
            'Li':[[3.460272990838495e21, 85.51998980232813e0, 2.1770706138013733e0],\
                  [3.6365138642921554e20, 4493.713340713575e0, 0.6964625952167312e0],\
                  [2.0659396971422902e22, 9256.32561931876e0, 0.9999745128918196e0]],\
            'C': [[1.8809310466441196e20, 258.1083104918942e0, 1.1984440550739555e0],\
                  [5.509352426405289e23, 822.523017116882e0, 1.987881643259267e0],\
                  [4.572825176241475e25, 2263.6763526780164e0, 2.198152910400111e0],\
                  [9.835564484116034e27, 3537.9525592618193e0, 2.674447703635764e0],\
                  [7.536123777733801e22, 53034.28979013096e0, 0.8628040191262976e0],\
                  [6.587972799911127e23, 74090.46603762524e0, 0.9996157827449552e0]],\
            'N': [[6.441568477283914e19, 378.4956025896395e0, 0.9350660865589433e0],\
                  [1.4701035287257558e23, 1100.1258100528623e0, 1.711847679118431e0],\
                  [4.963404971976876e25, 2232.374603536868e0, 2.2130314611564854e0],\
                  [1.429510866064162e27, 4658.081063546513e0, 2.3525381223431663e0],\
                  [1.3012849307544791e29, 6615.849778853221e0, 2.7281285044063783e0],\
                  [2.1409091975134166e23, 88606.4475724123e0, 0.8838466900060211e0],\
                  [1.4212916247536375e24, 117682.27130021283e0, 0.9994495038042972e0]],\
            'O': [[8.471015773053202e19, 343.2810702373722e0, 0.9990920676510977e0],\
                  [4.659951179950614e22, 1421.770921017062e0, 1.4896379815554006e0],\
                  [1.375943467286961e25, 2781.3610873184325e0, 1.985966133301126e0],\
                  [1.4410401568574448e27, 4652.670876621719e0, 2.353837077475246e0],\
                  [2.1812887948673846e28, 8303.41171218499e0, 2.4562134926799617e0],\
                  [1.0859737168339012e30, 11088.09514588445e0, 2.766300924086769e0],\
                  [5.135265231597041e23, 137319.4144041063e0, 0.8991975448040144e0],\
                  [2.7649244731504574e24, 175715.8649323391e0, 0.999259077262769e0]],\
            'Ne':[[1.2380498982127483e19, 684.0495717312216e0, 0.5886207199036224e0],\
                  [1.6861743227006676e22, 1790.87087951275e0, 1.3052851455522987e0],\
                  [4.012462791033353e24, 3450.2505445538854e0, 1.7789909006599105e0],\
                  [1.4251452589514637e26, 6544.999998027671e0, 1.9932260907792414e0],\
                  [6.670617838240815e27, 9689.667572133209e0, 2.282840577038655e0],\
                  [1.9391656057538128e29, 13557.847041347119e0, 2.522116807457813e0],\
                  [1.3881927947789378e30, 20383.797283570835e0, 2.586898525643846e0],\
                  [3.1075134088626144e31, 25254.466329809107e0, 2.8167466774677434e0],\
                  [2.1020658703689154e24, 282468.0585535283e0, 0.9200040384519763e0],\
                  [8.391062111505151e24, 343429.9456106982e0, 0.9988031600227598e0]],\
            'Ar':[[4.583155091154178e19, 427.3616288942586e0, 0.858307468844768e0],\
                  [2.347069522730368e23, 992.0665907024354e0, 1.8069357289301329e0],\
                  [1.931637188507353e26, 1775.9423286421918e0, 2.4675897958480384e0],\
                  [2.3004409750295724e28, 3141.43462246603e0, 2.8229627269683206e0],\
                  [3.468592152886133e30, 4422.602894394151e0, 3.263766733070865e0],\
                  [2.965675081118506e32, 5958.158993507227e0, 3.6326551045811497e0],\
                  [2.1249901067576396e33, 9479.178143644607e0, 3.629746649292696e0],\
                  [8.991061313075681e34, 11737.03857874398e0, 3.92742350307885e0],\
                  [7.611312705906995e29, 59342.85460567607e0, 2.229751529947494e0],\
                  [6.602179339057654e30, 71781.6325368281e0, 2.3680482459080237e0],\
                  [4.815600903407338e31, 85819.59318777094e0, 2.490706079578982e0],\
                  [2.055113377686043e32, 105186.65111959413e0, 2.558310025830369e0],\
                  [1.1945429165015238e33, 122591.38628416909e0, 2.663021333332675e0],\
                  [6.212435859768618e33, 141745.69057711778e0, 2.7584389081812857e0],\
                  [1.6060852787486324e34, 170916.68217921766e0, 2.783373249974532e0],\
                  [9.327897444360857e34, 190110.7378894357e0, 2.894937691775023e0],\
                  [6.6878740700780965e25, 1.8068758775178685e6, 0.9536895901987263e0],\
                  [1.5247916822996254e26, 2.0115330349032478e6, 0.9959340925415332e0]]}

class element_ion:
    def __init__(self, ADK_parameters_from='osiris', laser_wavelength =0.8e-6, element_name = 'H', ion_level = 0, max_level = None, E0_TV_m_array = np.arange(1,10,0.01), plot_line_color=None):
        # ADK_parameterrs_from can be either 'osiris' or 'fbpic'
        # This should only be set at initialization
        self._ADK_parameters_from = ADK_parameters_from
        self.element_name=element_name
        self.ion_level=ion_level
        self.E0_TV_m_array=E0_TV_m_array
        self.max_ion_level = max_level
        self.plot_line_color=plot_line_color
        #if self.plot_line_color is None, use ranbow colors
        self.colors = cm.rainbow(np.linspace(0, 1, self.max_ion_level))
        #d_phase is the intigral interval
        self.d_phase=np.pi/1024
        #laser omega in unit of s^-1, for 0.8 um wavelength laser
        self.laser_omega=c*2*np.pi/laser_wavelength
        #phace in 1/4 cycle
        self.phase_array = np.arange(self.d_phase, np.pi/2, self.d_phase)
        #integral interval dt is multiplied by 4,
        #effectively the value is transformed from 1/4 cycle to the whole cycle.
        self.integral_dt = 4.*self.d_phase/self.laser_omega

################################ Property ADK_parameters_from ################################
    def get_ADK_parameters_from(self):
        return self._ADK_parameters_from
    def set_ADK_parameters_from(self, value):
        print('Do not set ADK_parameters_from! It is only set when the object is initialized. Otherwise there is risk of errors!')
    ADK_parameters_from = property(get_ADK_parameters_from, set_ADK_parameters_from)

################################ Property element_name ################################
    def get_element_name(self):
        return self._element_name
    def set_element_name(self, value):
        self._element_name = value
        if self.ADK_parameters_from == 'osiris':
            self._ADK_parameters = osiris_ionization_parameters[value]
        else:
            ADK = ADK_calculator.ADK_calculator(element = value)
            self._ADK_parameters = ADK.get_ADK_parameters()
        self._max_ion_level=len(self._ADK_parameters)
    element_name = property(get_element_name, set_element_name)

################################ Property ADK_parameters ################################
    def get_ADK_parameters(self):
        return self._ADK_parameters
    def set_ADK_parameters(self, value):
        print('Warning: ADK_parameters cannot be set manually! Doing nothing.')
    ADK_parameters = property(get_ADK_parameters, set_ADK_parameters)

################################ Property max_ion_level ################################
    def get_max_ion_level(self):
        return self._max_ion_level
    def set_max_ion_level(self, value):
        if isinstance(value, int) and len(self._ADK_parameters) > value:
            self._max_ion_level = value
        else:
            if value is not None:
                print('Warning: Error while setting max_ion_level. Doing nothing.')
    max_ion_level = property(get_max_ion_level, set_max_ion_level)

################################method instant_E_GV_m_matrix################################
    def instant_E_GV_m_matrix(self):
        '''return instant E-field in GV/m.
           E0_TV_m_array: peak E-field in TV/m in an array.'''
        #do matrix product
        return np.matmul(1.e3*(self.E0_TV_m_array.reshape(self.E0_TV_m_array.size,1)),\
                            np.sin(self.phase_array).reshape(1,self.phase_array.size))

################################method instant_E_V_m_matrix################################
    def instant_E_V_m_matrix(self):
        '''return instant E-field in V/m.
           E0_TV_m_array: peak E-field in TV/m in an array.'''
        #do matrix product
        return np.matmul(1.e12*(self.E0_TV_m_array.reshape(self.E0_TV_m_array.size,1)),\
                            np.sin(self.phase_array).reshape(1,self.phase_array.size))

################################method ionization_rate_matrix################################
    def ionization_rate_matrix(self):
        '''return ionization rate matrix.'''
        # Get rate parameters
        rp=self.ADK_parameters[self.ion_level]
        if self.ADK_parameters_from == 'osiris':
            E_matrix=self.instant_E_GV_m_matrix()
            return rp[0]*E_matrix**(-rp[2])*np.exp(-rp[1]/E_matrix)
        else:
            # Calculate ionization rate as fbpic
            E_matrix=self.instant_E_V_m_matrix()
            return rp[0]*E_matrix**(rp[2])*np.exp(rp[1]/E_matrix)

################################method P_cycle_ion################################
    def P_cycle_ion(self):
        '''return ionization probabilities during one laser cycle.'''
        #phace in 1/4 cycle
        phase_array = np.arange(self.d_phase, np.pi/2, self.d_phase)
        ionization_rates = self.ionization_rate_matrix()
        #integrate with the trapezoidal rule. 
        intigral = np.trapz(ionization_rates, dx=self.integral_dt)
        return 1.-np.exp(-intigral)

################################method plot_P_ion################################
    def plot_P_ion(self, h_fig=None, h_ax=None, linestyle='-'):
        '''plot ionization probability within one cycle vs. E0.'''
        if h_fig is None:
            try: h_fig=self.h_fig
            except AttributeError: h_fig = self.h_fig = plt.figure()
        if h_ax is None:
            try: h_ax=self.h_ax
            except AttributeError: h_ax = self.h_ax = h_fig.add_subplot(111)
        if self.plot_line_color is None:
            line_color=self.colors[self.ion_level]
        else:
            line_color=self.plot_line_color
        label_ion_state = ''
        if 1==self.ion_level: label_ion_state = '$^+$'
        elif self.ion_level>1: label_ion_state = '$^{{{}+}}$'.format(self.ion_level)
        h_ax.plot(self.E0_TV_m_array, self.P_cycle_ion(), color=line_color, linestyle=linestyle, label='{}{}'.format(self.element_name, label_ion_state))
        h_ax.set_xlabel('$E_0\\ \\rm[TV/m]$')
        h_ax.set_ylabel('$P_{\\rm ion}$')
        return h_fig, h_ax

################################method plot_all_levels################################
    def plot_all_levels(self, h_fig=None, h_ax=None, linestyle='-'):
        for self.ion_level in range(self.max_ion_level): h_fig, h_ax = self.plot_P_ion(h_fig=h_fig, h_ax=h_ax, linestyle=linestyle)
        return h_fig, h_ax

if __name__ == '__main__':
    spec=element_ion(ADK_parameters_from = 'fbpic', element_name = 'F', laser_wavelength = 0.8e-6, ion_level = 0, E0_TV_m_array=np.arange(0.01,30.,0.01));
    h_fig, h_ax = spec.plot_all_levels()
    plt.legend()
    plt.tight_layout()

    #plt.show()
    plt.savefig('ion_prob_fb_{}.pdf'.format(spec.element_name))

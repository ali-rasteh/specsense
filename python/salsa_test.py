from backend import *
from backend import be_np as np, be_scp as scipy
from salsa import SALSA_Comp




class Params_Class(object):
    def __init__(self):

        self.fs = 983.04e6              # Input sample rate per antenna
        self.n_cc = 4                   # Number of component carriers
        self.fcc = 122.88e6             # Sample rate per component carrier fCC
        self.n_rx = 32                  # Number RX antennas, 32-1024
        self.n_tx = 32                  # Number TX antennas, 32-1024
        self.l_filt = 32                # Number of filter taps in up- and down-sampling
        self.n_filt_stages = 3          # Number of filter stages in up- and down-sampling
        self.us_rate = 2                # Up-sampling ratio
        self.ds_rate = 2                # Down-sampling ratio
        self.n_fft = 1024               # Number of FFT points
        self.n_sym_sf = 14              # Number of symbols per subframe
        self.t_sf = 125e-6              # Subframe duration
        self.n_sc_rb = 12               # Number of subcarriers per resource block
        self.n_rb = 69                  # Number of resource blocks
        self.n_sc = 828                 # Number of used subcarriers


        self.calc_params()


    def calc_params(self):

        self.t_cc = 1 / self.fcc                # Component carrier sample duration
        self.t_sym = self.t_sf / self.n_sym_sf  # Symbol duration
        self.f_sym = 1 / self.t_sym             # Symbol rate
        self.f_sc = self.fcc / self.n_fft       # Subcarrier spacing
        self.n_sym = self.t_sym / self.t_cc  # Number of symbols per component carrier
        self.n_cp = self.n_sym - self.n_fft
        print(self.n_cp)



if __name__ == "__main__":
    params = Params_Class()

    # Test the SALSA_Comp class
    salsa_comp = SALSA_Comp(params)
    # salsa_comp.test()
from backend import *
from backend import be_np as np, be_scp as scipy
from salsa import SALSA_Comp
from salsa_rt import SALSA_RT_alt
from SigProc_Comm.general import General



class Params_Class_Default(General):
    def __init__(self):
        super().__init__()
        
        # TODO: Complete the parameters

        self.seed = 42
        self.fs = 983.04e6              # Input sample rate per antenna
        self.fc = 2.6e9
        self.n_cc = 4                   # Number of component carriers
        self.fcc = 122.88e6             # Sample rate per component carrier fCC
        self.n_rx_ant = 32              # Number RX antennas, 32-1024
        self.n_tx_ant = 32              # Number TX antennas, 32-1024
        self.n_fft = 1024               # Number of FFT points
        self.n_sym_sf = 14              # Number of symbols per subframe
        self.t_sf = 125e-6              # Subframe duration
        self.n_sc_rb = 12               # Number of subcarriers per resource block
        self.n_rb = 69                  # Number of resource blocks
        self.n_sc = 828                 # Number of used subcarriers
        self.n_rs_rb = 4                # Number of reference signals per resource block TODO
        
        self.l_filt = 32                # Number of filter taps in up- and down-sampling
        self.l_match_filt = 32          # Number of filter taps in matched filter
        self.n_filt_stages = 3          # Number of filter stages in up- and down-sampling
        self.us_rate = 2                # Up-sampling ratio
        self.ds_rate = 2                # Down-sampling ratio


        # self.gnb_pos = None
        self.gnb_height_above_ground = 40
        self.ue_height_above_ground = 1.5
        # self.dist_range =np.array([20, 1000])
        self.dist_range =np.array([20, 100])
        # self.speed_range = np.array([0, 5])  # m/s
        self.speed_range = np.array([0, 0])  # m/s
        self.control_rx_power = True
        
        self.nue = 8
        self.n_gnb_sect = 1
        self.nrow_gnb = 8
        self.ncol_gnb = 4
        self.nrow_ue = 1
        self.ncol_ue = 1
        self.delay_spread = 100e-9
        self.scs = 120.0e3
        self.n_guard_carriers = [0, 0]
        self.dc_null = False
        self.pilot_pattern = "kronecker"
        self.pilot_ofdm_symbol_indices = [2, 11]
        self.cyclic_prefix_length = 20
        self.perfect_csi = False
        self.direction = "uplink"
        self.domain = "freq"
        self.n_ofdm_symbols = 14
        self.n_bits_per_symbol = 2
        self.coderate = 0.5
        self.freq_spacing = 'rb'
        self.ptx_ue_max_db = 26
        # self.snr_tgt_range_db = None
        self.gnb_nf = 2
        self.empty_scene = False
        self.n_cir_dataset = 5
        self.total_sim_duration = 10e-3
        self.batch_size = 1
        self.load_cir_dataset = False
        self.compute_ber = False
        self.normalize_channel = False
        self.channel_add_awgn = False
        
        self.verbose_level = 0
        self.plot_level = 0




    def calc_params(self):

        self.ov_samp_rate = self.fs / (self.fcc * self.n_cc)            # Oversampling rate

        self.t_cc = 1 / self.fcc                                        # Component carrier sample duration
        self.scs = self.fcc / self.n_fft                               # Subcarrier spacing
        self.used_bw = self.n_sc * self.scs                            # Used bandwidth

        self.t_sym_total = self.t_sf / self.n_sym_sf                    # OFDM Symbol duration including CP
        self.t_sym = 1/self.scs                                        # OFDM Symbol duration excluding CP
        self.t_cp = self.t_sym_total - self.t_sym                       # Cyclic prefix duration
        self.n_sym_per_sec = 1/self.t_sf * self.n_sym_sf                # Number of symbols per second

        # self.n_sym_total = self.t_sf / self.n_sym_sf * self.fcc       # Number of symbols per component carrier
        self.n_sym_total = self.t_sym_total / self.t_cc                 # Number of symbols per component carrier
        # self.n_cp = self.n_sym_total - self.n_fft                     # Number of samples in CP
        self.n_cp = self.t_cp / self.t_sym * self.n_fft                 # Number of samples in CP

        self.n_rs = self.n_rb * self.n_rs_rb                            # Average number of reference signals TODO
        
        self.delay_resolution = self.t_sym/self.n_sc
        self.doppler_resolution = self.scs/self.n_ofdm_symbols




class Params_Class(Params_Class_Default):
    def __init__(self):
        super().__init__()

        self.calc_params()
        self.print_params()





if __name__ == "__main__":
    params = Params_Class()
    
    salsa_rt_sim = SALSA_RT_alt(params)
    salsa_rt_sim.set_seed(seed=params.seed, to_set=["numpy", "tensorflow", "sionna"])
    salsa_rt_sim.run_simulation()
    
    # Test the SALSA_Comp class
    # salsa_comp = SALSA_Comp(params)
    # salsa_comp.test()




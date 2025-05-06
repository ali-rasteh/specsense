from backend import *
from backend import be_np as np, be_scp as scipy




class Model(Block):
    """This block simulates OFDM MIMO transmissions over the CDL model.

    Simulates point-to-point transmissions between a UT and a BS.
    Uplink and downlink transmissions can be realized with either perfect CSI
    or channel estimation. ZF Precoding for downlink transmissions is assumed.
    The receiver (in both uplink and downlink) applies LS channel estimation
    and LMMSE MIMO equalization. A 5G LDPC code as well as QAM modulation are
    used.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], tf.float32
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], tf.float32
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                scenario = "umi",
                domain = "freq",
                direction = "uplink",
                cdl_model = "A",
                channel_model = None,
                delay_spread = 100e-9,
                perfect_csi = True,
                speed = 0.0,
                cyclic_prefix_length = 6,
                pilot_ofdm_symbol_indices = [2, 11],
                subcarrier_spacing = 15e3
                ):
        super().__init__()

        # Provided parameters
        self._scenario = scenario
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._channel_model = channel_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut = 1
        # Must be a multiple of two if dual-polarized antennas are used
        self._num_ut_ant_row = 2
        self._num_ut_ant_col = 2
        self._num_ut_ant = self._num_ut_ant_row * self._num_ut_ant_col
        self._num_bs_ant_row = 8
        self._num_bs_ant_col = 4
        self._num_bs_ant = self._num_bs_ant_row * self._num_bs_ant_col
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 2
        self._coderate = 0.5

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant

        # Required system components
        # self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        # Configure antenna arrays
        self._ut_array = AntennaArray(num_rows=int(self._num_ut_ant_row/2),
                                      num_cols=int(self._num_ut_ant_col),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)
        
        # self._ut_array = AntennaArray(
        #                          num_rows=self._num_ut_ant_row,
        #                          num_cols=self._num_ut_ant_col,
        #                          polarization="single",
        #                          polarization_type="V",
        #                          antenna_pattern="omni",
        #                          carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=int(self._num_bs_ant_row/2),
                                      num_cols=int(self._num_bs_ant_col),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)


        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

        # self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
        #                                  normalize_channel=True, return_channel=True)



    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_ut,
                                self._scenario,
                                min_ut_velocity=0.0,
                                max_ut_velocity=0.0)

        self._channel_model.set_topology(*topology)


    @tf.function # Run in graph mode. See the following guide: https://www.tensorflow.org/guide/function
    def call(self, batch_size, ebno_db):
        
        # self.new_topology(batch_size)

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, self._num_tx, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations

            a, tau = self._cdl(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            y = self._channel_freq(x_rg, h_freq, no)
            # y, h = self._ofdm_channel(x_rg, no)

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est (y, no)

        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return b, b_hat
    






if __name__ == "__main__":
    
    UL_SIMS = {
        "ebno_db" : list(np.arange(-5, 20, 4.0)),
        "cdl_model" : ["A", "B", "E"],
        "delay_spread" : 100e-9,
        "domain" : "freq",
        "direction" : "uplink",
        "perfect_csi" : False,
        "speed" : 0.0,
        "cyclic_prefix_length" : 6,
        "pilot_ofdm_symbol_indices" : [2, 11],
        "ber" : [],
        "bler" : [],
        "duration" : None
    }

    start = time.time()

    for cdl_model in UL_SIMS["cdl_model"]:

        model = Model(domain=UL_SIMS["domain"],
                    direction=UL_SIMS["direction"],
                    cdl_model=cdl_model,
                    delay_spread=UL_SIMS["delay_spread"],
                    perfect_csi=UL_SIMS["perfect_csi"],
                    speed=UL_SIMS["speed"],
                    cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                    pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])

        ber, bler = sim_ber(model,
                            UL_SIMS["ebno_db"],
                            batch_size=256,
                            max_mc_iter=100,
                            num_target_block_errors=1000,
                            target_bler=1e-3)

        UL_SIMS["ber"].append(list(ber.numpy()))
        UL_SIMS["bler"].append(list(bler.numpy()))

    UL_SIMS["duration"] = time.time() - start









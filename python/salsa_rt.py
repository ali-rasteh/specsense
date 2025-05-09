# mimo_sim.py:  Simulation of the long-term beamforming and short-term spatial equalization for a gNB with multiple UEs
from backend import *
from backend import be_np as np, be_scp as scipy
from SigProc_Comm.general import General
from salsa_mimo import MIMO_OFDM, CIRGenerator
from miutils import CoverageMapPlanner




class SALSA_RT(General):
    """
    Main simulation object that performs one drop of the simulation:
     * A gNB is placed in the region 
     * A number of UEs are placed in the region
     * Channels are computed between the gNB and the UEs
    
     TODO: 
     * Compute the MU-MIMO post-equalization SNR.  
     First, we should compute the baseline capacity assuming the channel is perfectly known and we can perform
     the complete equalization.  
     * Create a loop so that we can run multiple drops of the UEs and then aggregate the channels
    """
    def __init__(self,
                 gnb_pos : np.ndarray = None, 
                 gnb_height_above_ground: float = 40,
                 ue_height_above_ground: float = 1.5,
                 dist_range : np.ndarray =None, 
                 nue : int = 8, 
                 n_gnb_sect : int = 3, 
                 nrow_gnb : int = 8,
                 ncol_gnb : int = 4,
                 nrow_ue : int = 1, 
                 ncol_ue : int = 1,
                 fc : float = 2.6e9,
                 delay_spread : float = 100e-9,
                 scs : float = 120.0e3,
                 n_guard_carriers : list = [0, 0],
                 dc_null : bool = False,
                 pilot_pattern : str = "kronecker",
                 pilot_ofdm_symbol_indices : list = [2, 11],
                 cyclic_prefix_length : int = 20,
                 perfect_csi : bool = False,
                 direction : str = "uplink",
                 domain : str = "freq",
                 n_ofdm_symbols : int = 14,
                 n_bits_per_symbol : int = 2,
                 n_sc_rb : int = 12,
                 n_rb : int = 69,
                 coderate : float = 0.5,
                #  bw : float = 100e6,
                 freq_spacing  : str = 'rb', 
                 ptx_ue_max : float = 26,
                 snr_tgt_range : np.ndarray | None = None,
                 gnb_nf : float = 2,
                 empty_scene : bool = False,
                 n_cir_dataset : int = 1,
                 batch_size : int = 1,
                 load_cir_dataset : bool = False,
                 compute_ber : bool = False,
                 normalize_channel : bool = False,
                 channel_add_awgn : bool = False):

        """
        Constructor 

        Parameters
        ----------
        gnb_pos : (3,) array-like
            Position of the gNB in the scene.  If None, 
            the gNB is placed in the center of the scene.
        dist_range : (2,) array-like
            Distance range of the gNB.  If None, the 
            distance range is set to [20, 2000] m. 
        nue : int
            Number of UEs per base station sector to be placed 
            in the scene.  
        n_gnb_sect : int 
            Number of sectors to be placed in the scene.
            Default is 3. 
        nrow_gnb : int
            Number of rows in the gNB array.  Default is 64.
        ncol_gnb : int
            Num of columns in the gNB array.  Default is 16.
        nrow_ue : int
            Number of rows in the UE array.  Default is 1.
        ncol_ue : int
            Number of columns in the UE array.  Default is 1.
        fc : float
            Carrier frequency in Hz.  Default is 2.6 GHz.
        scs : float
            Subcarrier spacing in Hz.  Default is 120 kHz.
        bw : float
            Bandwidth in Hz.  Default is 100 MHz.
        freq_spacing : str
            Frequency spacing.  Default is 'rb'.
            Options are 'rb' for resource block and 'sc' for subcarrier.
            If 'rb', the frequency spacing is set to the resource block size.
        snr_tgt_range : (2,) array-like or None
            SNR per antenna range in dB.  Default is [-6, 3].
            Ths is used to compute the uplink power control.  The power is ideally so that
            the RX SNR is snr_per_ant_range[1] dB at the gNB, but will be lower
            if the UE does not have enough power.  If the max power is below snr_per_ant_range[0] dB,
            the UE is considered in outage.
        n_cir_dataset : int
            Number of drops to perform.  Default is 1.
        load_cir_dataset : bool
            If True, the CIR dataset is loaded from a file.  If False, the CIR dataset is created
        """
        super().__init__()

        # Set the parameters
        self.n_cir_dataset = n_cir_dataset
        if gnb_pos is None:
            gnb_pos = np.array([0, 0])
        if dist_range is None:
            dist_range = np.array([20, 2000])
        self.gnb_pos = gnb_pos
        self.dist_range = dist_range
        self.nue = nue
        self.n_gnb_sect = n_gnb_sect
        self.gnb_height_above_ground = gnb_height_above_ground
        self.ue_height_above_ground = ue_height_above_ground
        self.nrow_gnb = nrow_gnb
        self.ncol_gnb = ncol_gnb
        self.nrow_ue = nrow_ue
        self.ncol_ue = ncol_ue
        self.fc = fc
        self.scs = scs
        self.load_cir_dataset = load_cir_dataset
        self.n_ofdm_symbols = n_ofdm_symbols
        self.freq_spacing = freq_spacing
        self.ptx_ue_max = ptx_ue_max # max TX power in dBm
        self.gnb_nf = gnb_nf  # gNB noise figure in dB
        if snr_tgt_range is None:
            # snr_tgt_range = np.array([-6, 3])
            snr_tgt_range = np.array([-6, 10])
        self.snr_tgt_range = snr_tgt_range
        self.n_bits_per_symbol = n_bits_per_symbol
        self.coderate = coderate
        self.n_guard_carriers = n_guard_carriers
        self.dc_null = dc_null
        self.pilot_pattern = pilot_pattern
        self.pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self.cyclic_prefix_length = cyclic_prefix_length
        self.perfect_csi = perfect_csi
        self.direction = direction
        self.domain = domain
        self.batch_size = batch_size
        self.delay_spread = delay_spread
        self.empty_scene = empty_scene
        self.n_sc_rb= n_sc_rb
        self.n_rb = n_rb
        self.compute_ber = compute_ber
        self.normalize_channel = normalize_channel
        self.channel_add_awgn = channel_add_awgn
        
        self.init()

        
        
    def init(self):
        # Compute the number of RBs
        # self.n_rb = int(self.used_bw / (self.scs * self.n_sc_rb))
        self.n_sc = self.n_rb * self.n_sc_rb
        self.used_bw = self.n_sc * self.scs

        self.nue_tot = self.nue*self.n_gnb_sect
        self.EkT = -174
        self.gnb_noise_fl_db = self.EkT + 10*np.log10(self.used_bw) + self.gnb_nf
        self.gnb_noise_fl = 10**(self.gnb_noise_fl_db/10)

        

    def load_scene(self):
        """
        Load the scene from a file or create a new scene.
        """
        # Load the scene
        if not self.empty_scene:
            mod_dir=os.path.dirname(os.path.abspath(__file__))
            scene_path=os.path.join(mod_dir, "Denver/denver.xml")
            self.scene = load_scene(scene_path)
        

            # Set a bounding box in the center bbox = [xmin, xmax, ymin, ymax]
            L_NS = 3700
            W_WE = 2900
            bbox = [-W_WE/2,W_WE/2,-L_NS/2,L_NS/2]

            # Create a coverage map
            self.grid_size = 10
            self.cm = CoverageMapPlanner(self.scene._scene, grid_size=self.grid_size)
            self.cm.set_grid()
            self.cm.compute_grid_attributes()

            # Place the gNB in the center of the scene
            self.dist = (self.cm.xgrid - self.gnb_pos[0])**2 +\
                (self.cm.ygrid - self.gnb_pos[1])**2
            idx = np.argmin(self.dist)
            tx_idx = np.unravel_index(idx, self.cm.xgrid.shape)
            self.gnb_pos = np.array(
                [self.cm.xgrid[tx_idx[0], tx_idx[1]],
                self.cm.ygrid[tx_idx[0], tx_idx[1]], 
                self.cm.zmax_grid[tx_idx[0], tx_idx[1]]])
            self.gnb_pos[2] += self.gnb_height_above_ground

            # Compute distances to all the grid points
            self.dist = np.sqrt((self.cm.xgrid - self.gnb_pos[0])**2 +\
                        (self.cm.ygrid - self.gnb_pos[1])**2 +\
                        (self.cm.zmax_grid - self.gnb_pos[2])**2)
            self.dist = np.sqrt(self.dist**2 + self.cm.zmin_grid**2)

            self.angles = np.arctan2(self.cm.ygrid - self.gnb_pos[1],
                self.cm.xgrid - self.gnb_pos[0])

        else:
            self.scene = sionna.rt.Scene()
            self.gnb_pos = np.array([0, 0, self.gnb_height_above_ground])





    def drop_users(self):
        """
        Randomly drop users, compute the channels and save the channels to a pickle file
        """
        
        self.ue_pos = np.zeros((self.nue_tot, 3))
        self.bs_ue_association = np.zeros([self.n_gnb_sect, self.nue_tot])
        
        # Add the gNB receivers.  There is one RX
        # per sector
        for s in range(self.n_gnb_sect):
            yaw = 2*np.pi*s / self.n_gnb_sect
            rx = sionna.rt.Receiver(
                    name=f"gnb-{s}",
                    position=self.gnb_pos, 
                    orientation=[yaw,0,0])
            self.scene.add(rx)

            # sect_orientation = np.array([np.cos(yaw), np.sin(yaw)])
            sect_angle_range = np.array([yaw - np.pi/self.n_gnb_sect, yaw + np.pi/self.n_gnb_sect])
            if np.min(sect_angle_range) < -np.pi:
                sect_angle_range += 2*np.pi
            if np.max(sect_angle_range) > np.pi:
                sect_angle_range -= 2*np.pi
            

            if not self.empty_scene:
                # Select nue points randomly that are within 
                # the distance range and not in a building
                idx = np.where(
                    (self.dist > self.dist_range[0])    &\
                    (self.dist < self.dist_range[1])    &\
                    (~self.cm.bldg_grid)                &\
                    (self.angles > sect_angle_range[0]) &\
                    (self.angles < sect_angle_range[1]))
                npts = len(idx[0])
                if npts < self.nue:
                    raise ValueError(f"Not enough points in the distance range {self.dist_range}.\n"
                                    f"Only {npts} points found.")
                I = np.random.choice(npts, self.nue, replace=False)
                ue_x = self.cm.xgrid[idx[0][I], idx[1][I]]
                ue_y = self.cm.ygrid[idx[0][I], idx[1][I]]
                ue_z = self.cm.zmax_grid[idx[0][I], idx[1][I]]
                self.ue_pos_sect = np.column_stack((ue_x, ue_y, ue_z))
                self.ue_pos_sect[:,2] += self.ue_height_above_ground

            else:
                r = np.random.uniform(self.dist_range[0], self.dist_range[1], size=(self.nue))
                phi = np.random.uniform(sect_angle_range[0], sect_angle_range[1], size=(self.nue))
                ue_x = r*np.cos(phi)
                ue_y = r*np.sin(phi)
                ue_z = self.ue_height_above_ground*np.ones(self.nue)
                self.ue_pos_sect = np.column_stack((ue_x, ue_y, ue_z))

            # Add the UE transmitters.  There is one TX
            # per UE
            for i in range(self.nue):
                ue_idx = s*self.nue + i
                tx = sionna.rt.Transmitter(name=f"ue-{ue_idx}",
                        color = [0.0, 1.0, 0.0],
                        position=self.ue_pos_sect[i])
                self.scene.add(tx)
                tx.look_at(self.gnb_pos)
                self.ue_pos[ue_idx] = self.ue_pos_sect[i]

                # distances = []
                # tx_to_gnb = self.ue_pos_sect[i, :2] - self.gnb_pos[:2]
                # tx_to_gnb_norm = tx_to_gnb / np.linalg.norm(tx_to_gnb)
                # distance = np.dot(sector_orientation, tx_to_gnb_norm)
                # distances.append(distance)

                # Find the sector with the maximum alignment
                # associated_sector = np.argmax(distances)
                self.bs_ue_association[s, ue_idx] = 1
            


        # Set the gNB array
        self.scene.rx_array = PlanarArray(
            num_rows=self.nrow_gnb, 
            num_cols=self.ncol_gnb,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V"
        )

        # Set the UE array
        self.scene.tx_array = PlanarArray(
            num_rows=self.nrow_ue, num_cols=self.ncol_ue,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V"
        )

        # Compute the paths
        # TODO:  I amde the samples_per_src and max_num_paths_per_src very small to speed up the simulation
        # Can we check this?
        solver = PathSolver()
        self.paths =  solver(
            scene=self.scene,
            max_depth=3, 
            samples_per_src=int(1e5),
            max_num_paths_per_src=100,
            synthetic_array=True
        )

        # Get the CIR.  We add a dimension to be compatible with the OFDM channel
        self.a, self.tau = self.paths.cir(out_type='tf')

        # Get the channel at the frequencies
        bw = self.scs*self.n_sc_rb 
        if self.freq_spacing == 'rb':
            self.freq = tf.linspace(-0.5, 0.5, self.n_rb)*bw
        elif self.freq_spacing == 'sc':  
            self.freq = tf.linspace(-0.5, 0.5, self.n_sc)*bw
        else:
            raise ValueError(f"freq_spacing must be 'rb' or 'sc'. Got {self.freq_spacing}.")
        tau = tf.expand_dims(self.tau, axis=0)
        a = tf.expand_dims(self.a, axis=0)
        self.chan = cir_to_ofdm_channel(self.freq, a, tau)

        # Remove first axis
        self.chan = tf.squeeze(self.chan, axis=(0,5))

        # Save the OFDM channel to a pickle file
        with open("chan.pkl", "wb") as f:
            pickle.dump([self.chan, self.ue_pos, self.gnb_pos], f)


    def load_users(self):
        """
        Load the channel from a pickle file
        """
        with open("chan.pkl", "rb") as f:
            self.chan, self.ue_pos, self.gnb_pos = pickle.load(f)


    def ue_associate_power_control(self):
        """
        Associate the UEs to the gNBs and set the power control
        """

        # Compute the channel gain to the strongest sector
        chan_gain = tf.reduce_mean(tf.abs(self.chan)**2, axis=(1,3,4)).numpy()
        chan_gain_max = np.max(chan_gain, axis=0)

        # Create a small value so that we don't have to deal with zeros
        self.chan_gain_max = np.maximum(chan_gain_max, 1e-30)
        # print(self.chan_gain_max)

        # Compute the SNR at the max TX power
        snr_max = self.ptx_ue_max + 10*np.log10(self.chan_gain_max) - self.gnb_noise_fl_db
        # print(f"snr_max: {snr_max}")
        
        # Find users that are in outage
        self.Iout = np.where(snr_max < self.snr_tgt_range[0])[0]
        self.Iconn = np.where(snr_max >= self.snr_tgt_range[0])[0]
        # print(f"Iconn: {self.Iconn}, Iout: {self.Iout}")

        # Find the power control level        
        pow_dec = np.maximum(snr_max - self.snr_tgt_range[1], 0)
        # print(f"pow_dec: {pow_dec}")

        # SNR after power control
        self.snr_avg = snr_max - pow_dec
        # print(self.snr_avg)
        self.snr_avg[self.Iout] = -np.inf

        # Compute the scaling on the channel so that they are scaled relative to a unit variance noise
        self.wvar = 1
        # self.chan_scale = np.zeros(self.nue_tot)
        self.chan_scale = np.ones(self.nue_tot)
        # print(self.chan_scale)
        # self.chan_scale[self.Iconn] = np.sqrt(10**(0.1*(self.snr_avg[self.Iconn]))/self.chan_gain_max[self.Iconn])
        self.chan_scale[self.Iconn] = np.sqrt(10**(0.1*(-pow_dec[self.Iconn])))
        # print(self.chan_scale)
        self.chan_scale[self.Iout] = 1.0
        # print(self.chan_scale)


    def compute_mimo_matrix(self):
        """
        Compute the MIMO matrix for the gNB and the UEs

        Computes

        H: (n_gnb_sect, nrx, nue, nue_tx, nfreq)
        """
        # Get the channel matrix for the connected UEs
        self.H = tf.gather(self.chan, self.Iconn, axis=2)
        
        # Scale the channel by the power control
        scale = self.chan_scale[self.Iconn]
        self.H = self.H * scale[None, None, :, None, None ]


    def compute_capacity_3gpp(self, snr):
        """
        Compute the channel capacity using the 3GPP model
        """
        alpha = 0.6
        beta = 1.0
        rho_max = 4.8

        if type(snr) is not np.ndarray:
            snr = np.array(snr)

        # Compute the capacity
        capacity = np.minimum(alpha * np.log2(1 + beta * snr), rho_max)

        return capacity
    

    def compute_baseline_capacity(self):
        """
        Compute the baseline capacity for the gNB
        """

        # Compute the capacity
        # H Shape: (nrx, nrx_ant, ntx, ntx_ant, nfreq)
        print("H Shape: ", self.H.shape)
        H_cap = tf.reduce_mean(tf.abs(self.H)**2, axis=(1,3,4)).numpy()
        max_idx = np.argmax(H_cap, axis=0)
        H_cap = np.zeros(self.H.shape[1:], dtype=self.H.numpy().dtype)
        for i in range(self.H.shape[2]):
            H_cap[:, i, :, :] = self.H[max_idx[i], :, i, :, :]
        # H_cap = np.maximum(H_cap, 1e-30)

        max_freq = np.argmax(np.mean(np.abs(H_cap)**2, axis=(0,1,2)))
        H_cap = H_cap[:, :, :, max_freq]

        caps = np.zeros(self.H.shape[2])
        for i in range(H_cap.shape[1]):
            H = H_cap[:, i, :]
            n_path = np.linalg.matrix_rank(H)
            alpha = 10**(0.1*self.ptx_ue_max) / n_path / self.gnb_noise_fl
            c = np.log2(np.linalg.det(np.eye(H.shape[1]) + alpha * np.abs(H.conj().T @ H)))
            # print(f"Capacity {i}: {c}")
        
        
        
    def plot_region(self):
        """
        Plot the region of interest in the scene
        """
        plt.figure(figsize=(10, 10))
        if not self.empty_scene:
            plt.imshow(self.cm.zmin_grid, interpolation='nearest',
                    extent = [self.cm.x[0], self.cm.x[-1], 
                                self.cm.y[0], self.cm.y[-1]])
            plt.plot(self.gnb_pos[0], self.gnb_pos[1], 'bo', markersize=10)
            plt.colorbar(label='Zmin (m)')


        # Plot the UEs not in outage
        plt.plot(self.ue_pos[self.Iconn,0], self.ue_pos[self.Iconn,1], 'go', markersize=10)
        plt.plot(self.ue_pos[self.Iout,0], self.ue_pos[self.Iout,1], 'rx', markersize=10)
        plt.xlabel("x (m)", fontsize=14)
        plt.ylabel("y (m)", fontsize=14)
        plt.title("Region of Interest", fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid()
        # plt.show()
        plt.savefig("region.png")

    

    def simulate_ray_tracing(self):

        a_list = []
        tau_list = []

        max_num_paths = 0

        n_run = self.n_cir_dataset
        for idx in range(n_run):
            print(f"Progress: {idx+1}/{n_run}", end="\r")

            self.load_scene()
            self.drop_users()
            # self.load_users()
            self.ue_associate_power_control()
            self.compute_mimo_matrix()

            a_list.append(self.a)
            tau_list.append(self.tau)

            # Update maximum number of paths over all batches of CIRs
            num_paths = self.a.shape[-2]
            if num_paths > max_num_paths:
                max_num_paths = num_paths
                    
            # self.plot_region()

        a = []
        tau = []
        for a_,tau_ in zip(a_list, tau_list):
            num_paths = a_.shape[-2]
            a_ = np.pad(a_, [[0,0],[0,0],[0,0],[0,0],[0,max_num_paths-num_paths],[0,0]], constant_values=0)
            tau_ = np.pad(tau_, [[0,0],[0,0],[0,max_num_paths-num_paths]], constant_values=0)
            a_ = np.expand_dims(a_, axis=0)
            tau_ = np.expand_dims(tau_, axis=0)
            a.append(a_)
            tau.append(tau_)
        
        a = np.concatenate(a, axis=0) # Concatenate along the num_rx dimension
        tau = np.concatenate(tau, axis=0)

        # # Add a batch_size dimension
        # a = np.expand_dims(a, axis=0)
        # tau = np.expand_dims(tau, axis=0)

        # # Exchange the num_tx and batchsize dimensions
        # a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
        # tau = np.transpose(tau, [2, 1, 0, 3])

        # Remove CIRs that have no active link (i.e., a is all-zero)
        p_link = np.sum(np.abs(a)**2, axis=tuple(range(1, a.ndim)))
        a = a[p_link>0.,...]
        tau = tau[p_link>0.,...]

        self.a = a
        self.tau = tau

        # Save self.a and self.tau for future use
        with open("cir_dataset.pkl", "wb") as f:
            pickle.dump({"a": self.a, "tau": self.tau, "bs_ue_association": self.bs_ue_association, "chan_scale": self.chan_scale}, f)


    def create_cir_dataset(self):
        
        n_dataset = self.a.shape[0]
        n_rx = self.a.shape[1]
        n_rx_ant = self.a.shape[2]
        n_tx = self.a.shape[3]
        n_tx_ant = self.a.shape[4]
        max_n_paths = self.a.shape[5]
        n_time_steps = self.a.shape[6]

        self.cir_generator = CIRGenerator(self.a, self.tau, self.chan_scale)

        # Initialises a channel model that can be directly used by OFDMChannel layer
        self.channel_model = CIRDataset(self.cir_generator,
                                self.batch_size,
                                n_rx,
                                n_rx_ant,
                                n_tx,
                                n_tx_ant,
                                max_n_paths,
                                n_time_steps)
        
        

    def run_simulation(self):
        # Run the simulation


        if self.load_cir_dataset:
            # Load the CIR dataset from a file
            with open("cir_dataset.pkl", "rb") as f:
                data = pickle.load(f)
                self.a = data["a"]
                self.tau = data["tau"]
                self.bs_ue_association = data["bs_ue_association"]
                self.chan_scale = data["chan_scale"]
        else:
            # Create the CIR dataset from the simulation
            self.simulate_ray_tracing()

        self.create_cir_dataset()

        self.n_sc = 128 * 3
        # self.nue_tot = 16
        
        self.phy_model = MIMO_OFDM(channel_mode = "dataset",
                domain = self.domain,
                direction = self.direction,
                channel_model = self.channel_model,
                normalize_channel = self.normalize_channel,
                channel_add_awgn = self.channel_add_awgn,
                delay_spread = self.delay_spread,
                perfect_csi = self.perfect_csi,
                # cyclic_prefix_length = self.cyclic_prefix_length,
                # pilot_ofdm_symbol_indices = self.pilot_ofdm_symbol_indices,
                # subcarrier_spacing = self.scs,
                carrier_frequency = self.fc,
                fft_size = self.n_sc,
                # num_ofdm_symbols = self.n_ofdm_symbols,
                num_sectors = self.n_gnb_sect,
                num_ut = self.nue_tot,
                bs_ut_association = self.bs_ue_association,
                num_ut_ant_row = self.nrow_ue,
                num_ut_ant_col = self.ncol_ue,
                num_bs_ant_row = self.nrow_gnb,
                num_bs_ant_col = self.ncol_gnb,
                dc_null = self.dc_null,
                num_guard_carriers = self.n_guard_carriers,
                # pilot_pattern = self.pilot_pattern,
                num_bits_per_symbol = self.n_bits_per_symbol,
                coderate = self.coderate)
        

        no = self.gnb_noise_fl
        b, b_hat = self.phy_model.call(batch_size=self.batch_size, no=no)
        x = self.phy_model.x.numpy()
        x_hat = self.phy_model.x_hat.numpy()

        
        # print(np.mean(np.abs(x)**2))
        # print(np.mean(np.abs(x_hat)**2))
        # print(np.mean(np.abs(x-x_hat)**2))
        # print(x_hat.shape)
        # print(x_hat[0,0,0,:20])
        post_eq_snr = 10*np.log10(np.mean(np.abs(x)**2)/np.mean(np.abs(x-x_hat)**2))
        print(f"Post-equalization SNR: {post_eq_snr} dB")
        # print(f"b shape: {b.shape}, b_hat shape: {b_hat.shape}")
        # print(b[0,0,0,:20])
        # print(b_hat[0,0,0,:20])
        # print(np.mean(np.abs(b-b_hat)**2))


        if self.compute_ber:
            # Compute the BER and BLER
            ebno_dbs=list(np.arange(-5, 20, 4.0))
            ber, bler = sim_ber(self.phy_model,
                                ebno_dbs=ebno_dbs,
                                batch_size=self.batch_size,
                                max_mc_iter=100,
                                num_target_block_errors=1000,
                                target_bler=1e-3)
            print(f"BER: {ber}, BLER: {bler}")




class SALSA_RT_alt(SALSA_RT):
    def __init__(self, params=None):
        super().__init__()
        
        self.import_attributes(params=params, overwrite=True)
        self.init()
        
        




if __name__ == "__main__":
    # Simulate paths
    sim = SALSA_RT()
    sim.run_simulation()




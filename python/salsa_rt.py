# mimo_sim.py:  Simulation of the long-term beamforming and short-term spatial equalization for a gNB with multiple UEs
import sionna
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import cir_to_ofdm_channel
import os
import pickle


# We reload the nonlin module to avoid re-running the code in it
# when we are in the interactive mode (e.g. Jupyter notebook).
import importlib
import sys
if 'miutils' not in sys.modules:
    import miutils 
else:
    importlib.reload(miutils)
from miutils import CoverageMapPlanner

class Sim(object):
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
                 dist_range : np.ndarray =None, 
                 nue : int = 4, 
                 nsect : int = 3, 
                 nrow_gnb : int = 8, 
                 ncol_gnb : int = 4,
                 scs_khz : float = 120., 
                 bw_mhz : float = 100,
                 freq_spacing  : str = 'rb', 
                 snr_tgt_range : np.ndarray | None = None):

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
        nsect : int 
            Number of sectors to be placed in the scene.
            Default is 3. 
        nrow_gnb : int
            Number of rows in the gNB array.  Default is 64.
        ncol_gnb : int
            Num of columns in the gNB array.  Default is 16.
        scs_khz : float
            Subcarrier spacing in kHz.  Default is 120 kHz.
        bw_mhz : float
            Bandwidth in MHz.  Default is 100 MHz.
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
        """

        # Set the parameters
        if gnb_pos is None:
            gnb_pos = np.array([0, 0])
        if dist_range is None:
            dist_range = np.array([20, 2000])
        self.gnb_pos = gnb_pos
        self.dist_range = dist_range
        self.nue = nue
        self.nsect = nsect
        self.gnb_height_above_ground = 40
        self.ue_height_above_ground = 1.5
        self.nrow_gnb = nrow_gnb
        self.ncol_gnb = ncol_gnb
        self.scs_khz = scs_khz
        self.bw_mhz = bw_mhz
        self.freq_spacing = freq_spacing
        self.ptx_ue_max = 26 # max TX power in dBm
        self.gnb_nf = 2  # gNB noise figure in dB
        if snr_tgt_range is None:
            snr_tgt_range = np.array([-6, 3])
        self.snr_tgt_range = snr_tgt_range

        self.nue_tot = self.nue*self.nsect
        

        # Compute the number of RBs
        self.nsc_rb= 12
        self.nrb = int(self.bw_mhz*1e3 / (self.scs_khz * self.nsc_rb))
        self.nsc = self.nrb * self.nsc_rb
        self.bw_mhz = self.nsc * self.scs_khz / 1e3
        
        
        

    def drop_users(self):
        """
        Randomly drop users, compute the channels and save the channels to a pickle file
        """
        # Load the scene
        self.empty_scene = False
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
            dist = (self.cm.xgrid - self.gnb_pos[0])**2 +\
                (self.cm.ygrid - self.gnb_pos[1])**2
            idx = np.argmin(dist)
            tx_idx = np.unravel_index(idx, self.cm.xgrid.shape)
            self.gnb_pos = np.array(
                [self.cm.xgrid[tx_idx[0], tx_idx[1]],
                self.cm.ygrid[tx_idx[0], tx_idx[1]], 
                self.cm.zmax_grid[tx_idx[0], tx_idx[1]]])
            self.gnb_pos[2] += self.gnb_height_above_ground

            # Compute distances to all the grid points
            dist = np.sqrt((self.cm.xgrid - self.gnb_pos[0])**2 +\
                        (self.cm.ygrid - self.gnb_pos[1])**2 +\
                        (self.cm.zmax_grid - self.gnb_pos[2])**2)
            dist = np.sqrt(dist**2 + self.cm.zmin_grid**2)

            # Select nue points randomly that are within 
            # the distance range and not in a building
            idx = np.where(
                (dist > self.dist_range[0]) &\
                (dist < self.dist_range[1]) &\
                (~self.cm.bldg_grid))
            npts = len(idx[0])
            if npts < self.nue:
                raise ValueError(f"Not enough points in the distance range {self.dist_range}.\n"
                                f"Only {npts} points found.")
            I = np.random.choice(npts, self.nue*self.nsect, replace=False)
            ue_x = self.cm.xgrid[idx[0][I], idx[1][I]]
            ue_y = self.cm.ygrid[idx[0][I], idx[1][I]]
            ue_z = self.cm.zmax_grid[idx[0][I], idx[1][I]]
            self.ue_pos = np.column_stack((ue_x, ue_y, ue_z))
            self.ue_pos[:,2] += self.ue_height_above_ground

        else:
            self.scene = sionna.rt.Scene()
            self.gnb_pos = np.array([0, 0, self.gnb_height_above_ground])
            
            r = np.random.uniform(self.dist_range[0], self.dist_range[1], size=(self.nue*self.nsect))
            phi = np.random.uniform(0, 2*np.pi, size=(self.nue*self.nsect))
            ue_x = r*np.cos(phi)
            ue_y = r*np.sin(phi)
            ue_z = self.ue_height_above_ground*np.ones(self.nue*self.nsect)
            self.ue_pos = np.column_stack((ue_x, ue_y, ue_z))


        
        # Add the gNB receivers.  There is one RX
        # per sector
        for s in range(self.nsect):
            yaw = 2*np.pi*s / self.nsect
            rx = sionna.rt.Receiver(
                    name=f"gnb-{s}",
                    position=self.gnb_pos, 
                    orientation=[yaw,0,0])
            self.scene.add(rx)

        # Add the UE transmitters.  There is one TX
        # per UE
        for i in range(self.nue*self.nsect):
            rx = sionna.rt.Transmitter(name=f"ue-{i}",
                        color = [0.0, 1.0, 0.0],
                        position=self.ue_pos[i])
            self.scene.add(rx)
            rx.look_at(self.gnb_pos)

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
            num_rows=1, num_cols=1,
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
            max_num_paths_per_src=10,
            synthetic_array=True
        )

        # Get the CIR.  We add a dimension to be compatible with the OFDM channel
        self.a, self.tau = self.paths.cir(out_type='tf')
        self.a = tf.expand_dims(self.a, axis=0)
        self.tau = tf.expand_dims(self.tau, axis=0)

        # Get the channel at the frequencies
        bw = self.scs_khz*1e3*self.nsc_rb 
        if self.freq_spacing == 'rb':
            self.freq = tf.linspace(-0.5, 0.5, self.nrb)*bw
        elif self.freq_spacing == 'sc':  
            self.freq = tf.linspace(-0.5, 0.5, self.nsc)*bw
        else:
            raise ValueError(f"freq_spacing must be 'rb' or 'sc'.  Got {self.freq_spacing}.")
        self.chan = cir_to_ofdm_channel(self.freq, self.a, self.tau)

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

        # Compute the SNR at the max TX power
        EkT = -174
        snr_max = self.ptx_ue_max + 10*np.log10(self.chan_gain_max) -\
              self.gnb_nf - EkT -10*np.log10(self.bw_mhz*1e6)
        
        # Find users that are in outage
        self.Iout = np.where(snr_max < self.snr_tgt_range[0])[0]
        self.Iconn = np.where(snr_max >= self.snr_tgt_range[0])[0]

        # Find the power control level        
        pow_dec = np.maximum(snr_max - self.snr_tgt_range[1], 0)

        # SNR after power control
        self.snr_avg = snr_max - pow_dec
        self.snr_avg[self.Iout] = -np.inf

        # Compute the scaling on the channel so that they are scaled relative to a unit variance noise
        self.wvar = 1
        self.chan_scale = np.zeros(self.nue_tot)
        self.chan_scale[self.Iconn] = np.sqrt(10**(0.1*(self.snr_avg[self.Iconn]))/self.chan_gain_max[self.Iconn])



    def compute_mimo_matrix(self):
        """
        Compute the MIMO matrix for the gNB and the UEs

        Computes

        H: (nsect, nrx, nue, nue_tx, nfreq)
        """
        # Get the channel matrix for the connected UEs
        self.H = tf.gather(self.chan, self.Iconn, axis=2)
        
        # Scale the channel by the power control
        scale = self.chan_scale[self.Iconn]
        self.H  = self.H  * scale[None, None, :, None, None ]


    def compute_3gpp_capacity(self, snr):
        """
        Compute the 3GPP capacity for the gNB
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
        plt.show()
        plt.savefig("region.png")

    

# Simulate paths
drop_users = True
sim = Sim(nue=10,dist_range=np.array([20, 1000]))
if drop_users:
    sim.drop_users()
sim.load_users()
sim.ue_associate_power_control()
sim.compute_mimo_matrix()
sim.compute_baseline_capacity()
sim.plot_region()

# Find the SNR to 


    # Save the paths with pickle
    #with open("paths.pkl", "wb") as f:
    #    pickle.dump(sim.paths, f)

# Load the paths with pickle
#with open("paths.pkl", "rb") as f:
#    paths = pickle.load(f)


from backend import *
from backend import be_np as np, be_scp as scipy
from SigProc_Comm.general import General
from enum import Enum


import sionna
import tensorflow as tf
# import sionnautils
import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import cir_to_ofdm_channel
import os

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
    def __init__(self, gnb_pos=None, dist_range=None, nue = 16, 
                 nsect = 3, nrow_gnb = 64, ncol_gnb = 16,
                 scs_khz = 120, bw_mhz = 100):

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
        

        TODO:  Add descriptions of the other parameters
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


        # Compute the number of RBs
        self.nsc_rb= 12
        self.nrb = int(self.bw_mhz*1e3 / (self.scs_khz * self.nsc_rb))
        self.nsc = self.nrb * self.nsc_rb
        
        # Load the scene
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
        
        # Add the gNB receivers.  There is one RX
        # per sector
        for s in range(nsect):
            yaw = 2*np.pi*s / nsect
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
            samples_per_src=1000,
            max_num_paths_per_src=10,
            synthetic_array=True
        )

        # Get the CIR.  We add a dimension to be compatible with the OFDM channel
        self.a, self.tau = self.paths.cir(out_type='tf')
        self.a = tf.expand_dims(self.a, axis=0)
        self.tau = tf.expand_dims(self.tau, axis=0)

        # Get the channel at the frequencies
        self.freq = tf.linspace(-0.5, 0.5, self.nsc)*self.scs_khz*1e3*self.nsc
        self.chan = cir_to_ofdm_channel(self.freq, self.a, self.tau)
        
        



    def plot_region(self):
        """
        Plot the region of interest in the scene
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.cm.zmin_grid, interpolation='nearest',
                   extent = [self.cm.x[0], self.cm.x[-1], 
                             self.cm.y[0], self.cm.y[-1]],)
        plt.plot(self.gnb_pos[0], self.gnb_pos[1], 'ro', markersize=5)
        plt.plot(self.ue_pos[:,0], self.ue_pos[:,1], 'bo', markersize=5)
        plt.colorbar()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.grid()
        plt.show()

    

# Simulate paths
get_paths = True
import pickle

if get_paths:
    sim = Sim()
    sim.plot_region()

    # Save the paths with pickle
    #with open("paths.pkl", "wb") as f:
    #    pickle.dump(sim.paths, f)

# Load the paths with pickle
#with open("paths.pkl", "rb") as f:
#    paths = pickle.load(f)


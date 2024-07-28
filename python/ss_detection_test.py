import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand
from ss_detection import *
from ss_detection_Unet import *
import argparse
import os
import sys



class params_class(object):
    def __init__(self):
        
        self.n_fft=1024
        self.n_samples=1024
        self.shape=(self.n_fft,)
        self.n_sigs_min=0
        self.n_sigs_max=10
        self.snr_min=0.5
        self.snr_max=100
        self.sig_size_min=1
        self.sig_size_max=100
        self.noise_power=1
        self.n_simulations=10
        self.sweep_steps=10

        self.n_dataset=100000
        self.generate_dataset=True
        self.dataset_path='./data/psd_dataset.npz'

        self.eval_smooth=1e-6
        self.train_ratio=0.8
        self.val_ratio=0.0
        self.test_ratio=0.2
        self.seed=42
        self.batch_size=64
        self.n_layers=10
        self.sched_gamma=0.1
        self.sched_step_size=10
        self.nepoch_save=10
        self.nbatch_log=400
        self.model_save_path='./model/'

        self.lr=1e-2
        self.n_epochs=10
        self.norm_mode='max'
        self.train=False
        self.test=True
        self.load_model_params=True
        self.model_load_path='./model/model_weights_10.pth'

        self.snrs = np.logspace(np.log10(self.snr_min), np.log10(self.snr_max), self.sweep_steps)
        self.sizes = np.logspace(np.log10(self.sig_size_min), np.log10(self.sig_size_max), self.sweep_steps)
        self.sizes = self.sizes.astype(int)




if __name__ == '__main__':

    params = params_class()

    ss_det = specsense_detection(params)
    if params.generate_dataset:
        ss_det.generate_psd_dataset(shape=(params.n_dataset, params.n_fft,), n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, sig_size_min=(params.sig_size_min,), sig_size_max=(params.sig_size_max,), snrs=params.snrs)


    ss_det_unet = ss_detection_Unet(params)
    ss_det_unet.generate_model()
    ss_det_unet.load_model()
    ss_det_unet.load_optimizer()
    ss_det_unet.generate_data_loaders()
    ss_det_unet.train_model()
    ss_det_unet.test_model()


    det_rate_snrs = ss_det.sweep_snrs(snrs=params.snrs, n_sigs=1, sig_size_min=(20,), sig_size_max=(20,))
    print(det_rate_snrs)
    det_rate_sizes = ss_det.sweep_sizes(sizes=params.sizes, n_sigs=1, snrs=np.array([10]))
    print(det_rate_sizes)



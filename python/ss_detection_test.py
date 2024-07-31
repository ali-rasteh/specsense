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
        self.n_samples=80
        self.shape=(self.n_fft,)
        self.sig_size_min=(1,)
        self.sig_size_max=(100,)
        self.n_sigs_min=0
        self.n_sigs_max=1
        self.snr_min=0.5
        self.snr_max=100
        self.noise_power=1
        self.sweep=True
        self.n_simulations=100
        self.sweep_steps=20
        self.sw_fixed_size=5
        self.sw_fixed_snr=5
        self.sw_sig_size_min=(1,)
        self.sw_sig_size_max=(100,)
        self.sw_n_sigs_min=0
        self.sw_n_sigs_max=1
        self.sw_snr_min=0.5
        self.sw_snr_max=100

        self.n_dataset=100000
        self.generate_dataset=False
        self.mask_mode='binary'        # binary or snr or channels
        self.norm_mode_data='std'        # max or std or max&std or none
        self.norm_mode_mask='none'        # max or std or max&std or none
        self.dataset_path='./data/psd_dataset.npz'

        self.lr=1e-2
        self.n_epochs=50
        self.train=False
        self.test=False
        self.load_model_params=False
        self.model_load_path='./model/model_weights_20.pth'

        self.eval_smooth=1e-6
        self.train_ratio=0.8
        self.val_ratio=0.0
        self.test_ratio=0.2
        self.seed=50
        self.batch_size=64
        self.n_layers=10
        self.sched_gamma=0.1
        self.sched_step_size=10
        self.nepoch_save=10
        self.nbatch_log=400
        self.model_save_path='./model/'



        self.snrs = np.logspace(np.log10(self.snr_min), np.log10(self.snr_max), self.sweep_steps)
        seen = set()
        self.snrs = [x for x in self.snrs if not (x in seen or seen.add(x))]
        
        self.sw_snrs = np.logspace(np.log10(self.sw_snr_min), np.log10(self.sw_snr_max), self.sweep_steps)
        seen = set()
        self.sw_snrs = [x for x in self.sw_snrs if not (x in seen or seen.add(x))]

        sizes_vals = [np.logspace(np.log10(self.sig_size_min[i]), np.log10(self.sig_size_max[i]), self.sweep_steps).astype(int) for i in range(len(self.shape))]
        self.sizes = [tuple(x) for x in zip(*sizes_vals)]
        seen = set()
        self.sizes = [x for x in self.sizes if not (x in seen or seen.add(x))]

        sw_sizes_vals = [np.logspace(np.log10(self.sw_sig_size_min[i]), np.log10(self.sw_sig_size_max[i]), self.sweep_steps).astype(int) for i in range(len(self.shape))]
        self.sw_sizes = [tuple(x) for x in zip(*sw_sizes_vals)]
        seen = set()
        self.sw_sizes = [x for x in self.sw_sizes if not (x in seen or seen.add(x))]

        self.fixed_snr_t = np.array(self.sw_fixed_snr)
        self.fixed_size_t = tuple([self.sw_fixed_size for _ in range(len(self.shape))])

if __name__ == '__main__':

    params = params_class()
    for attr in dir(params):
        if not callable(getattr(params, attr)) and not attr.startswith("__"):
            print(f"{attr} = {getattr(params, attr)}")

    ss_det = specsense_detection(params)
    X = np.random.rand(3,3)  # Example 3D array
    # LLRs = ss_det.compute_llrs(X)
    # ss_det.ML_detector_efficient(X)
    # ss_det.compute_interval(X)
    if params.generate_dataset:
        ss_det.generate_psd_dataset(dataset_path=params.dataset_path, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snrs=params.snrs, mask_mode=params.mask_mode)


    if params.train or params.test:
        ss_det_unet = ss_detection_Unet(params)
        ss_det_unet.generate_model()
        ss_det_unet.load_model()
        ss_det_unet.load_optimizer()
        ss_det_unet.generate_data_loaders()
        ss_det_unet.train_model()
        ss_det_unet.test_model()


    if params.sweep and params.test:
        det_rate_snrs = {}
        for snr in params.sw_snrs:
            dataset_path='./data/psd_dataset_snr-{:0.2f}'.format(snr)+'.npz'
            ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, sig_size_min=params.fixed_size_t, sig_size_max=params.fixed_size_t, snrs=np.array([snr]), mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate_snrs[snr] = ss_det_unet.test_acc
        print("NN detection rate for SNRs: {}".format(det_rate_snrs))
        # det_rate_snrs={0.5: 0.031520095648787404, 0.9008241153272055: 0.1534920985492083, 1.622968173510085: 0.4491734054332343, 2.9240177382128656: 0.7636754817475145, 5.268051384453321: 0.9115785185140542, 9.491175455796848: 0.9617008338340174, 17.09975946676696: 0.9776975720073469, 30.807751387916714: 0.9876163278144008, 55.504730778481125: 0.9895433973961364, 100.0: 0.9893144725229793}


        det_rate_sizes = {}
        for size in params.sw_sizes:
            dataset_path='./data/psd_dataset_size-{}'.format(str(size).replace(" ", ""))+'.npz'
            ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, sig_size_min=size, sig_size_max=size, snrs=params.fixed_snr_t, mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate_sizes[size] = ss_det_unet.test_acc
        print("NN detection rate for signal sizes: {}".format(det_rate_sizes))


    if params.sweep:
        det_rate_snrs = ss_det.sweep_snrs(snrs=params.sw_snrs, n_sigs_min=1, n_sigs_max=1, sig_size_min=params.fixed_size_t, sig_size_max=params.fixed_size_t)
        print("ML detection rate for SNRs: {}".format(det_rate_snrs))
        # det_rate_snrs= {0.01: 0.0, 0.027825594022071243: 0.0, 0.0774263682681127: 0.0, 0.21544346900318834: 0.0, 0.5994842503189409: 0.0, 1.6681005372000592: 0.2916666666666667, 4.6415888336127775: 0.8976785714285712, 12.915496650148826: 0.9342424242424242, 35.93813663804626: 0.9799999999999999, 100.0: 0.9899999999999999}


        det_rate_sizes = ss_det.sweep_sizes(sizes=params.sw_sizes, n_sigs_min=1, n_sigs_max=1, snrs=params.fixed_snr_t)
        print("ML detection rate for signal sizes: {}".format(det_rate_sizes))
        # det_rate_sizes= {(1,): 0.0, (2,): 0.55, (4,): 0.38608168360977346, (7,): 0.7642857142857142, (12,): 0.9038095238095237, (21,): 0.9318354978354977, (35,): 0.9503699303699303, (59,): 0.9833842734092805, (100,): 0.9772513253587801}




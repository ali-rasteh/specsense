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
        
        self.n_fft=128
        self.n_samples=128
        self.shape=(self.n_fft, self.n_samples)
        # self.shape=(self.n_fft,)

        self.sig_size_min=(16,16)
        self.sig_size_max=(16,16)
        self.sw_fixed_size=16
        self.sw_sig_size_min=(1,1)
        self.sw_sig_size_max=(256,256)
        self.size_sam_mode='log'        # lin or log
        self.snr_min=0.5
        self.snr_max=100.0
        self.sw_fixed_snr=2.0
        self.sw_snr_min=0.5
        self.sw_snr_max=100.0
        self.snr_sam_mode='log'        # lin or log
        self.n_sigs_min=0
        self.n_sigs_max=1
        self.sw_n_sigs_min=0
        self.sw_n_sigs_max=1
        
        self.sweep_snr=True
        self.sweep_size=False
        self.n_simulations=50
        self.sweep_steps=20

        self.n_dataset=200000
        self.generate_dataset=True
        self.remove_dataset=True
        self.mask_mode='binary'        # binary or snr or channels
        self.norm_mode_data='std'        # max or std or max&std or none
        self.norm_mode_mask='none'        # max or std or max&std or none

        self.lr=1e-2
        self.n_epochs=30
        self.apply_pos_weight=False
        self.mask_thr=0.0
        self.draw_histogram=False
        self.train=False
        self.test=True
        self.load_model_params=True
        self.model_name='model_weights_30.pth'

        self.noise_power=1.0
        self.ML_thr_coeff = 1.5
        self.ML_thr=self.ML_thr_coeff*10.0*self.noise_power
        self.eval_smooth=1e-6
        self.train_ratio=0.8
        self.val_ratio=0.0
        self.test_ratio=0.2
        self.seed=50
        self.batch_size=64
        self.n_layers=7
        self.sched_gamma=0.1
        self.sched_step_size=10
        self.nepoch_save=10
        self.nbatch_log=400
        self.hist_thr=10.0
        self.hist_bins=40
        self.model_dir='./model/'
        self.figs_dir='./figs/'
        self.logs_dir='./logs/'
        self.data_dir='./data/'
        self.dataset_name='psd_dataset.npz'


        self.snr_range = np.array([self.snr_min, self.snr_max]).astype(float)
        
        self.sw_snrs = np.logspace(np.log10(self.sw_snr_min), np.log10(self.sw_snr_max), self.sweep_steps)
        seen = set()
        self.sw_snrs = [x for x in self.sw_snrs if not (x in seen or seen.add(x))]
        self.sw_snrs = np.array(self.sw_snrs).astype(float)

        sw_sizes_vals = [np.logspace(np.log10(self.sw_sig_size_min[i]), np.log10(self.sw_sig_size_max[i]), self.sweep_steps).astype(int) for i in range(len(self.shape))]
        self.sw_sizes = [tuple(x) for x in zip(*sw_sizes_vals)]
        seen = set()
        self.sw_sizes = [x for x in self.sw_sizes if not (x in seen or seen.add(x))]

        self.fixed_snr_range = np.array([self.sw_fixed_snr, self.sw_fixed_snr]).astype(float)
        self.fixed_size = tuple([self.sw_fixed_size for _ in range(len(self.shape))])


        for path in [self.model_dir, self.data_dir, self.figs_dir, self.logs_dir]:
            if not os.path.exists(path):
                os.makedirs(path)



if __name__ == '__main__':

    params = params_class()
    for attr in dir(params):
        if not callable(getattr(params, attr)) and not attr.startswith("__"):
            print(f"{attr} = {getattr(params, attr)}")

    ss_det = specsense_detection(params)
    ss_det.plot_MD_vs_SNR()
    ss_det.plot_MD_vs_DoF()
    print("Random string for this run: {}".format(ss_det.gen_random_str()))
    if params.generate_dataset:
        ss_det.generate_psd_dataset(dataset_path=params.data_dir+params.dataset_name, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snr_range=params.snr_range, mask_mode=params.mask_mode)


    if params.train or params.test:
        ss_det_unet = ss_detection_Unet(params)
        ss_det_unet.generate_data_loaders()
        ss_det_unet.generate_model()
        ss_det_unet.load_model()
        ss_det_unet.load_optimizer()
        ss_det_unet.train_model()
        ss_det_unet.test_model()

    if params.remove_dataset and os.path.exists(params.data_dir+params.dataset_name):
        os.remove(params.data_dir+params.dataset_name)


    det_rate_snrs = {}
    det_rate_sizes = {}
    if params.sweep_snr and params.test:
        det_rate = {}
        for snr in params.sw_snrs:
            dataset_path=params.data_dir+'psd_dataset_snr-{:0.2f}'.format(snr)+'.npz'
            if params.generate_dataset:
                ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, sig_size_min=params.fixed_size, sig_size_max=params.fixed_size, snr_range=np.array([snr,snr]), mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate[snr] = ss_det_unet.test_acc
            if params.remove_dataset and os.path.exists(dataset_path):
                os.remove(dataset_path)
        print("NN detection rate for SNRs: {}".format(det_rate))
        det_rate_snrs['snr_NN'] = det_rate.copy()

    if params.sweep_size and params.test:
        det_rate = {}
        for size in params.sw_sizes:
            dataset_path=params.data_dir+'psd_dataset_size-{}'.format(str(size).replace(" ", ""))+'.npz'
            if params.generate_dataset:
                ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, sig_size_min=size, sig_size_max=size, snr_range=params.fixed_snr_range, mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate[size] = ss_det_unet.test_acc
            if params.remove_dataset and os.path.exists(dataset_path):
                os.remove(dataset_path)
        print("NN detection rate for signal sizes: {}".format(det_rate))
        det_rate_sizes['size_NN'] = det_rate.copy()


    if params.sweep_snr:
        params.ML_thr = ss_det.find_ML_thr(thr_coeff=params.ML_thr_coeff)
        det_rate = ss_det.sweep_snrs(snrs=params.sw_snrs, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, sig_size_min=params.fixed_size, sig_size_max=params.fixed_size)
        print("ML detection rate for SNRs: {}".format(det_rate))
        det_rate_snrs['snr_ML'] = det_rate.copy()

    if params.sweep_size:
        det_rate = ss_det.sweep_sizes(sizes=params.sw_sizes, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, snr_range=params.fixed_snr_range)
        print("ML detection rate for signal sizes: {}".format(det_rate))
        det_rate_sizes['size_ML'] = det_rate.copy()


    # det_rate_snrs['snr_NN'] = {0.5: 0.00329078077173823, 0.6608103647168022: 0.010168008535419435, 0.8733406762343062: 0.03251702495724497, 1.1542251415688212: 0.08433338158130646, 1.5254478735307908: 0.1954936855137348, 2.0160635313287045: 0.35691675617694857, 2.6644713548591303: 0.5480446282386779, 3.5214205755638686: 0.7315886569023132, 4.653982429719222: 0.8578291773796082, 6.150799653536695: 0.9292866907119751, 8.129024324707132: 0.9589469064712525, 10.74348705760295: 0.9687766605377197, 14.198815201729696: 0.9761167923927307, 18.765448504002958: 0.9820855812072754, 24.800805740009125: 0.9858380811691284, 32.77725897265199: 0.9882502233505249, 43.319104912270475: 0.9907844389915467, 57.25142703256576: 0.9923591924667359, 75.66467275589427: 0.9930261597633362, 100.0: 0.9931096076965332}
    # det_rate_sizes['size_NN'] = {(1,): 0.005061195744653047, (2,): 0.022283786251449653, (3,): 0.05467798238992691, (4,): 0.10808066050410271, (5,): 0.1804213700771332, (7,): 0.3483198815822601, (10,): 0.562700422668457, (13,): 0.6936767197608947, (18,): 0.8032814272880554, (24,): 0.8588570240974426, (33,): 0.8996480465888977, (44,): 0.9263599509239197, (59,): 0.9473758228302002, (79,): 0.9623713345527649, (106,): 0.9731439858436585, (142,): 0.9799017094612121, (191,): 0.9832067944526672, (256,): 0.9823403435707092}
    # det_rate_snrs['snr_ML'] = {0.5: 0.0005, 0.6608103647168022: 0.0085, 0.8733406762343062: 0.05020917471503943, 1.1542251415688212: 0.18482047759527712, 1.5254478735307908: 0.5173705482176436, 2.0160635313287045: 0.8086030956357834, 2.6644713548591303: 0.8729467450972735, 3.5214205755638686: 0.9267670623067751, 4.653982429719222: 0.9309104142234573, 6.150799653536695: 0.9532519762845847, 8.129024324707132: 0.955919521805917, 10.74348705760295: 0.9681103896103895, 14.198815201729696: 0.976348484848485, 18.765448504002958: 0.9785238095238096, 24.800805740009125: 0.9872857142857145, 32.77725897265199: 0.9864037267080747, 43.319104912270475: 0.9905714285714287, 57.25142703256576: 0.9917337662337667, 75.66467275589427: 0.995071428571429, 100.0: 0.9960714285714292}
    # det_rate_sizes['size_ML'] = {(1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0075, (5,): 0.0, (7,): 0.05830128205128205, (10,): 0.19539793539793546, (13,): 0.21997448048597426, (18,): 0.47818304223744723, (24,): 0.7435307446661368, (33,): 0.8897729185120934, (44,): 0.9282022750425882, (59,): 0.9422651951653213, (79,): 0.9559282061772282, (106,): 0.9653049450759268, (142,): 0.9750333068426427, (191,): 0.9798842378658682, (256,): 0.986466015034174}
    # ss_det.plot(plot_dic={**det_rate_snrs, **det_rate_sizes}, snrs=params.sw_snrs, sizes=params.sw_sizes)
    if params.sweep_snr:
        ss_det.plot(plot_dic=det_rate_snrs, snrs=params.sw_snrs)
    if params.sweep_size:
        ss_det.plot(plot_dic=det_rate_sizes, sizes=params.sw_sizes)


import numpy as np
from ss_detection import specsense_detection
from ss_detection_Unet import ss_detection_Unet
import argparse
import os



class params_class(object):
    def __init__(self):
        
        self.n_fft=1024
        self.n_samples=128
        # self.shape=(self.n_fft, self.n_samples)
        self.shape=(self.n_fft,)

        self.sig_size_min=(4,4)
        self.sig_size_max=(256,256)
        self.sw_fixed_size=30
        self.sw_sig_size_min=(1,1)
        self.sw_sig_size_max=(256,256)
        self.size_sam_mode='log'        # lin or log
        self.snr_min=1.0
        self.snr_max=100.0
        self.sw_fixed_snr=10.0
        self.sw_snr_min=0.5
        self.sw_snr_max=100.0
        self.snr_sam_mode='log'        # lin or log
        self.n_sigs_min=0
        self.n_sigs_max=1
        self.n_sigs_p_dist=[0.1,0.9]
        # self.n_sigs_p_dist=None
        self.sw_n_sigs_min=1
        self.sw_n_sigs_max=1
        self.sw_n_sigs_p_dist=None
        
        self.sweep_snr=['nn','ml']     # nn or ml
        self.sweep_size=['nn','ml']    # nn or ml
        self.n_simulations=100
        self.sweep_steps=20
        self.n_adj_search=1
        self.n_largest=3

        self.n_dataset=200000
        self.generate_dataset=True
        self.remove_dataset=True
        self.mask_mode='binary'        # binary or snr or channels
        self.norm_mode_data='std'        # max or std or max&std or none
        self.norm_mode_mask='none'        # max or std or max&std or none
        self.norm_mode_bbox='len'        # max or std or max&std or none or len

        self.lr=1e-2
        self.n_epochs_tot=100
        self.n_epochs_unet=50
        self.train=True
        self.test=True
        self.load_model_params=[]        # List of model parameters to load, unet and model
        self.save_model=True
        self.model_name='cO2pk9_weights_10.pth'
        self.model_unet_name='cO2pk9_unet_weights_10.pth'
        self.problem_mode='detection'     # segmentation or detection
        self.det_mode='nn-unet'     # nn-unet or nn-simple or contours
        self.train_mode='separate'     # end2end or separate
        self.obj_det_loss_mode='mse'    # iou or mse
        self.contours_min_area=1
        self.contours_max_gap=1
        self.lambda_start=10.0
        self.lambda_length=2.0
        self.lambda_obj=1.0
        self.lambda_class=1.0

        # Constant parameters
        self.n_epochs_dethead=self.n_epochs_tot-self.n_epochs_unet
        self.draw_histogram=False
        self.mask_thr=0.0
        self.gt_mask_thr=0.5
        self.apply_pos_weight=False
        self.noise_power=1.0
        self.ML_thr_coeff=1.5
        self.ML_thr=self.ML_thr_coeff*10.0*self.noise_power
        self.eval_smooth=1e-6
        self.train_ratio=0.8
        self.val_ratio=0.0
        self.test_ratio=0.2
        self.seed=50
        self.batch_size=64
        self.n_layers=int(np.log2(min(self.shape)).round())
        self.sched_gamma=0.1
        self.sched_step_size=10
        self.nepoch_save=10
        self.nbatch_log=400
        self.hist_thr=10.0
        self.hist_bins=40
        self.test_n_dataset=10000
        self.random_str=None
        self.ML_mode='torch'       # np or torch
        self.model_save_dir='./model/'
        self.model_load_dir='./model/backup/'
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

        self.n_sigs_p_dist = np.array(self.n_sigs_p_dist) if self.n_sigs_p_dist is not None else self.n_sigs_p_dist
        self.sw_n_sigs_p_dist = np.array(self.sw_n_sigs_p_dist) if self.sw_n_sigs_p_dist is not None else self.sw_n_sigs_p_dist

        for path in [self.model_save_dir, self.model_load_dir, self.data_dir, self.figs_dir, self.logs_dir]:
            if not os.path.exists(path):
                os.makedirs(path)



if __name__ == '__main__':

    params = params_class()
    print("Run parameters:")
    for attr in dir(params):
        if not callable(getattr(params, attr)) and not attr.startswith("__"):
            print(f"{attr} = {getattr(params, attr)}")
    print('\n')

    ss_det = specsense_detection(params)
    ss_det.plot_MD_vs_SNR()
    ss_det.plot_MD_vs_DoF()
    params.random_str = ss_det.gen_random_str()
    print("Random string for this run: {}".format(params.random_str))
    if params.generate_dataset:
        if params.train:
            ss_det.generate_psd_dataset(dataset_path=params.data_dir+params.dataset_name, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, n_sigs_p_dist=params.n_sigs_p_dist, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snr_range=params.snr_range, mask_mode=params.mask_mode)
        elif params.test:
            ss_det.generate_psd_dataset(dataset_path=params.data_dir+params.dataset_name, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, n_sigs_p_dist=params.n_sigs_p_dist, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snr_range=params.snr_range, mask_mode=params.mask_mode)


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
    if 'nn' in params.sweep_snr and params.test:
        det_rate = {}
        for snr in params.sw_snrs:
            dataset_path=params.data_dir+'psd_dataset_snr-{:0.2f}'.format(snr)+'.npz'
            if params.generate_dataset:
                ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=params.fixed_size, sig_size_max=params.fixed_size, snr_range=np.array([snr,snr]), mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.train_ratio=0.01
            ss_det_unet.test_ratio=0.99
            ss_det_unet.val_ratio=0.0
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate[snr] = ss_det_unet.test_acc
            if params.remove_dataset and os.path.exists(dataset_path):
                os.remove(dataset_path)
        print("NN detection rate for SNRs: {}\n".format(det_rate))
        det_rate_snrs['snr_NN'] = det_rate.copy()

    if 'nn' in params.sweep_size and params.test:
        det_rate = {}
        for size in params.sw_sizes:
            dataset_path=params.data_dir+'psd_dataset_size-{}'.format(str(size).replace(" ", ""))+'.npz'
            if params.generate_dataset:
                ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=size, sig_size_max=size, snr_range=params.fixed_snr_range, mask_mode=params.mask_mode)
            ss_det_unet.dataset_path=dataset_path
            ss_det_unet.train_ratio=0.01
            ss_det_unet.test_ratio=0.99
            ss_det_unet.val_ratio=0.0
            ss_det_unet.generate_data_loaders()
            ss_det_unet.test_model()
            det_rate[size] = ss_det_unet.test_acc
            if params.remove_dataset and os.path.exists(dataset_path):
                os.remove(dataset_path)
        print("NN detection rate for signal sizes: {}\n".format(det_rate))
        det_rate_sizes['size_NN'] = det_rate.copy()


    if 'ml' in params.sweep_snr:
        params.ML_thr = ss_det.find_ML_thr(thr_coeff=params.ML_thr_coeff)
        det_rate = ss_det.sweep_snrs(snrs=params.sw_snrs, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=params.fixed_size, sig_size_max=params.fixed_size, mode='simple')
        print("ML detection rate for SNRs: {}\n".format(det_rate))
        det_rate_snrs['snr_ML'] = det_rate.copy()

        det_rate = ss_det.sweep_snrs(snrs=params.sw_snrs, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=params.fixed_size, sig_size_max=params.fixed_size, mode='binary')
        print("ML-binary search detection rate for SNRs: {}\n".format(det_rate))
        det_rate_snrs['snr_ML_binary_search'] = det_rate.copy()

    if 'ml' in params.sweep_size:
        det_rate = ss_det.sweep_sizes(sizes=params.sw_sizes, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, snr_range=params.fixed_snr_range, mode='simple')
        print("ML detection rate for signal sizes: {}\n".format(det_rate))
        det_rate_sizes['size_ML'] = det_rate.copy()

        det_rate = ss_det.sweep_sizes(sizes=params.sw_sizes, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, snr_range=params.fixed_snr_range, mode='binary')
        print("ML-binary search detection rate for signal sizes: {}\n".format(det_rate))
        det_rate_sizes['size_ML_binary_search'] = det_rate.copy()


    # det_rate_snrs['snr_NN'] = {0.5: 0.00013185336267108132, 0.6608103647168022: 0.0015672112395619337, 0.8733406762343062: 0.013371356799185668, 1.1542251415688212: 0.07687240992168713, 1.5254478735307908: 0.24722358791431434, 2.0160635313287045: 0.4892272824520826, 2.6644713548591303: 0.685144745282865, 3.5214205755638686: 0.8048406510052658, 4.653982429719222: 0.8767833832007426, 6.150799653536695: 0.9207975303991931, 8.129024324707132: 0.9500897710658429, 10.74348705760295: 0.9673707998434445, 14.198815201729696: 0.9787792107978044, 18.765448504002958: 0.9866002537129577, 24.800805740009125: 0.9914856492413843, 32.77725897265199: 0.9949680512864293, 43.319104912270475: 0.9971241964469626, 57.25142703256576: 0.9982189919336162, 75.66467275589427: 0.9988371361053048, 100.0: 0.9989291927348432}
    # det_rate_sizes['size_NN'] = {(1,): 0.005061195744653047, (2,): 0.022283786251449653, (3,): 0.05467798238992691, (4,): 0.10808066050410271, (5,): 0.1804213700771332, (7,): 0.3483198815822601, (10,): 0.562700422668457, (13,): 0.6936767197608947, (18,): 0.8032814272880554, (24,): 0.8588570240974426, (33,): 0.8996480465888977, (44,): 0.9263599509239197, (59,): 0.9473758228302002, (79,): 0.9623713345527649, (106,): 0.9731439858436585, (142,): 0.9799017094612121, (191,): 0.9832067944526672, (256,): 0.9823403435707092}
    # det_rate_snrs['snr_ML'] = {0.5: 0.0, 0.6608103647168022: 0.0, 0.8733406762343062: 0.20918055555555556, 1.1542251415688212: 0.6687270622895626, 1.5254478735307908: 0.9220299145299148, 2.0160635313287045: 0.951387939221273, 2.6644713548591303: 0.9756111111111115, 3.5214205755638686: 0.9897777777777783, 4.653982429719222: 1.0000000000000004, 6.150799653536695: 0.9977777777777783, 8.129024324707132: 0.9977777777777783, 10.74348705760295: 1.0000000000000004, 14.198815201729696: 1.0000000000000004, 18.765448504002958: 1.0000000000000004, 24.800805740009125: 1.0000000000000004, 32.77725897265199: 1.0000000000000004, 43.319104912270475: 1.0000000000000004, 57.25142703256576: 1.0000000000000004, 75.66467275589427: 1.0000000000000004, 100.0: 1.0000000000000004}
    # det_rate_sizes['size_ML'] = {(1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0075, (5,): 0.0, (7,): 0.05830128205128205, (10,): 0.19539793539793546, (13,): 0.21997448048597426, (18,): 0.47818304223744723, (24,): 0.7435307446661368, (33,): 0.8897729185120934, (44,): 0.9282022750425882, (59,): 0.9422651951653213, (79,): 0.9559282061772282, (106,): 0.9653049450759268, (142,): 0.9750333068426427, (191,): 0.9798842378658682, (256,): 0.986466015034174}
    # ss_det.plot(plot_dic={**det_rate_snrs, **det_rate_sizes})
    print(det_rate_snrs)
    print(det_rate_sizes)
    if params.sweep_snr:
        ss_det.plot(plot_dic=det_rate_snrs)
    if params.sweep_size:
        ss_det.plot(plot_dic=det_rate_sizes)


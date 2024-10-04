from backend import *
from backend import be_np as np, be_scp as scipy
from ss_detection import SS_Detection
from ss_detection_unet import SS_Detection_Unet

import_cupy=False
import_torch=True



class Params_Class(object):
    def __init__(self):
        
        self.n_fft=1024
        self.n_samples=128
        # self.shape=(self.n_fft, self.n_samples)
        self.shape=(self.n_fft,)

        self.sig_size_min=(1,1)
        self.sig_size_max=(256,256)
        self.sw_fixed_size_list=[8, 16, 32, 128]
        self.sw_sig_size_min=(1,1)
        self.sw_sig_size_max=(256,256)
        self.size_sam_mode='log'        # lin or log
        self.snr_min=0.5
        self.snr_max=100.0
        self.sw_fixed_snr_list=[2.0, 5.0, 10.0, 50.0]
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
        
        self.sweep_snr=['ml']     # nn and ml
        self.sweep_size=['ml']    # nn and ml
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

        self.train=False
        self.test=False
        self.n_epochs_tot=50
        self.n_epochs_seg=50
        self.lr=1e-2
        self.load_model_params=['model']        # List of model parameters to load, seg and model
        self.save_model=True
        self.model_name='ThjNRm_weights_50.pth'
        self.model_seg_name='ThjNRm_weights_50.pth'
        self.problem_mode='segmentation'     # segmentation or detection
        self.seg_mode='unet'     # unet or threshold
        self.det_mode='contours'     # nn_segnet or nn_features or contours (nn_features is with feature extraction using LLRs and a NN afterward)
        self.train_mode='end2end'     # end2end or separate
        self.obj_det_loss_mode='mse'    # iou or mse
        self.lambda_start=100.0
        self.lambda_length=20.0
        self.lambda_obj=1.0
        self.lambda_class=1.0
        self.contours_min_area=1
        self.contours_max_gap=20
        self.verbose_level=5
        self.plot_level=5

        # Constant parameters
        self.n_epochs_dethead=self.n_epochs_tot-self.n_epochs_seg
        self.draw_histogram=False
        self.mask_thr=0.0
        self.gt_mask_thr=0.5
        self.apply_pos_weight=False
        self.noise_power=1.0
        self.ML_thr_coeff=1.9
        self.ML_thr=self.ML_thr_coeff*10.0*self.noise_power
        self.ML_PFA=1e-6
        self.ML_thr_mode='static'    # analysis or data or static
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
        self.use_cupy=False
        self.use_torch=True
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

        self.n_sigs_p_dist = np.array(self.n_sigs_p_dist) if self.n_sigs_p_dist is not None else self.n_sigs_p_dist
        self.sw_n_sigs_p_dist = np.array(self.sw_n_sigs_p_dist) if self.sw_n_sigs_p_dist is not None else self.sw_n_sigs_p_dist

        self.import_cupy=import_cupy
        self.import_torch=import_torch

        for path in [self.model_save_dir, self.model_load_dir, self.data_dir, self.figs_dir, self.logs_dir]:
            if not os.path.exists(path):
                os.makedirs(path)



if __name__ == '__main__':

    params = Params_Class()
    ss_det = SS_Detection(params)
    params.random_str = ss_det.gen_random_str()
    ss_det.print_info(params)
    ss_det.plot_MD_vs_SNR()
    ss_det.plot_MD_vs_DoF(mode=1)
    ss_det.plot_MD_vs_DoF(mode=2)
    if params.generate_dataset:
        if params.train:
            ss_det.generate_psd_dataset(dataset_path=params.data_dir+params.dataset_name, n_dataset=params.n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, n_sigs_p_dist=params.n_sigs_p_dist, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snr_range=params.snr_range, mask_mode=params.mask_mode)
        elif params.test:
            ss_det.generate_psd_dataset(dataset_path=params.data_dir+params.dataset_name, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.n_sigs_min, n_sigs_max=params.n_sigs_max, n_sigs_p_dist=params.n_sigs_p_dist, sig_size_min=params.sig_size_min, sig_size_max=params.sig_size_max, snr_range=params.snr_range, mask_mode=params.mask_mode)


    if params.train or params.test:
        ss_det_unet = SS_Detection_Unet(params)
        ss_det_unet.generate_data_loaders()
        ss_det_unet.generate_model()
        ss_det_unet.load_model()
        ss_det_unet.load_optimizer()
        ss_det_unet.train_model()
        ss_det_unet.test_model()

    if params.remove_dataset and os.path.exists(params.data_dir+params.dataset_name):
        os.remove(params.data_dir+params.dataset_name)


    metrics = {"det_rate": {}, "missed_rate": {}, "fa_rate": {}}
    if 'nn' in params.sweep_snr and params.test:
        det_rate = {}
        for size in params.sw_fixed_size_list:
            det_rate[size] = {}
            fixed_size = tuple([size for _ in range(len(params.shape))])
            for snr in params.sw_snrs:
                dataset_path=params.data_dir+'psd_dataset_snr-{:0.2f}'.format(snr)+'.npz'
                if params.generate_dataset:
                    ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=fixed_size, sig_size_max=fixed_size, snr_range=np.array([snr,snr]), mask_mode=params.mask_mode)
                ss_det_unet.dataset_path=dataset_path
                ss_det_unet.train_ratio=0.01
                ss_det_unet.test_ratio=0.99
                ss_det_unet.val_ratio=0.0
                ss_det_unet.generate_data_loaders()
                ss_det_unet.test_model()
                det_rate[size][snr] = ss_det_unet.test_acc
                if params.remove_dataset and os.path.exists(dataset_path):
                    os.remove(dataset_path)
        ss_det.print("NN detection rate for SNRs: {}\n".format(det_rate),thr=0)
        metrics['det_rate']['snr_NN'] = det_rate.copy()

    if 'nn' in params.sweep_size and params.test:
        det_rate = {}
        for snr in params.sw_fixed_snr_list:
            det_rate[snr] = {}
            fixed_snr_range = np.array([snr, snr]).astype(float)
            for size in params.sw_sizes:
                dataset_path=params.data_dir+'psd_dataset_size-{}'.format(str(size).replace(" ", ""))+'.npz'
                if params.generate_dataset:
                    ss_det.generate_psd_dataset(dataset_path=dataset_path, n_dataset=params.test_n_dataset, shape=params.shape, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=size, sig_size_max=size, snr_range=fixed_snr_range, mask_mode=params.mask_mode)
                ss_det_unet.dataset_path=dataset_path
                ss_det_unet.train_ratio=0.01
                ss_det_unet.test_ratio=0.99
                ss_det_unet.val_ratio=0.0
                ss_det_unet.generate_data_loaders()
                ss_det_unet.test_model()
                det_rate[snr][size] = ss_det_unet.test_acc
                if params.remove_dataset and os.path.exists(dataset_path):
                    os.remove(dataset_path)
        ss_det.print("NN detection rate for signal sizes: {}\n".format(det_rate),thr=0)
        metrics['det_rate']['size_NN'] = det_rate.copy()

    if 'ml' in params.sweep_snr or 'ml' in params.sweep_size:
        params.ML_thr = ss_det.find_ML_thr(thr_coeff=params.ML_thr_coeff)

    if 'ml' in params.sweep_snr:
        for metric in metrics:
            metrics[metric]['snr_ML'] = {}
            metrics[metric]['snr_ML_binary_search'] = {}
        for size in params.sw_fixed_size_list:
            fixed_size = tuple([size for _ in range(len(params.shape))])

            sweep_metrics = ss_det.sweep_snrs(snrs=params.sw_snrs, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, sig_size_min=fixed_size, sig_size_max=fixed_size)
            ss_det.print("ML metrics for SNRs for size:{} : {}\n".format(size, {key:sweep_metrics[key]['ML'] for key in list(sweep_metrics.keys())}),thr=0)
            ss_det.print("ML-binary search metrics for SNRs for size:{} : {}\n".format(size, {key:sweep_metrics[key]['ML_binary_search'] for key in list(sweep_metrics.keys())}),thr=0)
            for metric in metrics:
                metrics[metric]['snr_ML'][size] = sweep_metrics[metric]['ML'].copy()
                metrics[metric]['snr_ML_binary_search'][size] = sweep_metrics[metric]['ML_binary_search'].copy()

    if 'ml' in params.sweep_size:
        for metric in metrics:
            metrics[metric]['size_ML'] = {}
            metrics[metric]['size_ML_binary_search'] = {}
        for snr in params.sw_fixed_snr_list:
            fixed_snr_range = np.array([snr, snr]).astype(float)

            sweep_metrics = ss_det.sweep_sizes(sizes=params.sw_sizes, n_sigs_min=params.sw_n_sigs_min, n_sigs_max=params.sw_n_sigs_max, n_sigs_p_dist=params.sw_n_sigs_p_dist, snr_range=fixed_snr_range)
            ss_det.print("ML metrics for signal sizes for snr: {} : {}\n".format(snr, {key:sweep_metrics[key]['ML'] for key in list(sweep_metrics.keys())}),thr=0)
            ss_det.print("ML-binary search metrics for signal sizes for snr: {} : {}\n".format(snr, {key:sweep_metrics[key]['ML_binary_search'] for key in list(sweep_metrics.keys())}),thr=0)
            for metric in metrics:
                metrics[metric]['size_ML'][snr] = sweep_metrics[metric]['ML'].copy()
                metrics[metric]['size_ML_binary_search'][snr] = sweep_metrics[metric]['ML_binary_search'].copy()


    # metrics['snr_NN'] = {0.5: 0.00013185336267108132, 0.6608103647168022: 0.0015672112395619337, 0.8733406762343062: 0.013371356799185668, 1.1542251415688212: 0.07687240992168713, 1.5254478735307908: 0.24722358791431434, 2.0160635313287045: 0.4892272824520826, 2.6644713548591303: 0.685144745282865, 3.5214205755638686: 0.8048406510052658, 4.653982429719222: 0.8767833832007426, 6.150799653536695: 0.9207975303991931, 8.129024324707132: 0.9500897710658429, 10.74348705760295: 0.9673707998434445, 14.198815201729696: 0.9787792107978044, 18.765448504002958: 0.9866002537129577, 24.800805740009125: 0.9914856492413843, 32.77725897265199: 0.9949680512864293, 43.319104912270475: 0.9971241964469626, 57.25142703256576: 0.9982189919336162, 75.66467275589427: 0.9988371361053048, 100.0: 0.9989291927348432}
    # metrics['size_NN'] = {(1,): 0.005061195744653047, (2,): 0.022283786251449653, (3,): 0.05467798238992691, (4,): 0.10808066050410271, (5,): 0.1804213700771332, (7,): 0.3483198815822601, (10,): 0.562700422668457, (13,): 0.6936767197608947, (18,): 0.8032814272880554, (24,): 0.8588570240974426, (33,): 0.8996480465888977, (44,): 0.9263599509239197, (59,): 0.9473758228302002, (79,): 0.9623713345527649, (106,): 0.9731439858436585, (142,): 0.9799017094612121, (191,): 0.9832067944526672, (256,): 0.9823403435707092}
    # metrics['snr_ML'] = {0.5: 0.0, 0.6608103647168022: 0.0, 0.8733406762343062: 0.20918055555555556, 1.1542251415688212: 0.6687270622895626, 1.5254478735307908: 0.9220299145299148, 2.0160635313287045: 0.951387939221273, 2.6644713548591303: 0.9756111111111115, 3.5214205755638686: 0.9897777777777783, 4.653982429719222: 1.0000000000000004, 6.150799653536695: 0.9977777777777783, 8.129024324707132: 0.9977777777777783, 10.74348705760295: 1.0000000000000004, 14.198815201729696: 1.0000000000000004, 18.765448504002958: 1.0000000000000004, 24.800805740009125: 1.0000000000000004, 32.77725897265199: 1.0000000000000004, 43.319104912270475: 1.0000000000000004, 57.25142703256576: 1.0000000000000004, 75.66467275589427: 1.0000000000000004, 100.0: 1.0000000000000004}
    # metrics['size_ML'] = {(1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0075, (5,): 0.0, (7,): 0.05830128205128205, (10,): 0.19539793539793546, (13,): 0.21997448048597426, (18,): 0.47818304223744723, (24,): 0.7435307446661368, (33,): 0.8897729185120934, (44,): 0.9282022750425882, (59,): 0.9422651951653213, (79,): 0.9559282061772282, (106,): 0.9653049450759268, (142,): 0.9750333068426427, (191,): 0.9798842378658682, (256,): 0.986466015034174}
    
    ss_det.print(metrics,thr=0)
    if params.sweep_snr:
        ss_det.plot(plot_dic=metrics, mode='snr')
    if params.sweep_size:
        ss_det.plot(plot_dic=metrics, mode='size')




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand, randint, choice, exponential
from filter_utils import *
import time
import string
import random
import itertools
from scipy.stats import chi2



class specsense_detection(object):
    def __init__(self, params):

        self.shape = params.shape
        self.n_simulations = params.n_simulations
        self.noise_power = params.noise_power
        self.ML_thr = params.ML_thr
        self.figs_dir = params.figs_dir

        print("Initialized Spectrum Sensing class instance.")


    def plot_chi_squared_cdf(DoF=1, x_max=100, n_points=1000):
        # Generate x values
        x = np.linspace(0, x_max, n_points)
        
        # Calculate CDF values
        cdf = chi2.cdf(x, DoF)
        
        # Plot the CDF
        plt.figure(figsize=(8, 6))
        plt.plot(x, cdf, label=f'CDF of Chi-Squared (DoF={DoF})')
        plt.title('Cumulative Distribution Function of the Chi-Squared Distribution')
        plt.xlabel('x')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.show()


    def generate_random_regions(self, shape=(1000,), n_regions=1, min_size=None, max_size=None):
        regions = []
        ndims = len(shape)
        for _ in range(n_regions):
            region_slices = []
            for d, dim in enumerate(shape):
                if min_size is not None and max_size is not None:
                    size = randint(min_size[d], max_size[d] + 1)
                else:
                    size = randint(1, min(101, (dim+1)//2+1))
                start = randint(0, dim-size+1)
                size = min(size, dim-start)
                region_slices.append(slice(start, start + size))
            regions.append(tuple(region_slices))
        return regions


    def generate_random_PSD(self, shape=(1000,), sig_regions=None, n_sigs=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None, noise_power=1, snrs=np.array([10]), mask_mode='binary'):

        sig_powers = noise_power * snrs
        psd = exponential(noise_power, shape)
        if mask_mode=='binary' or mask_mode=='snr':
            mask = np.zeros(shape, dtype=float)
        elif mask_mode=='channels':
            mask = np.zeros((n_sigs_max,)+shape, dtype=float)

        if sig_regions is None:
            regions = self.generate_random_regions(shape=shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max)
        else:
            regions = sig_regions

        for sig_id, region in enumerate(regions):
            sig_power = choice(sig_powers)
            region_shape = tuple(slice_.stop - slice_.start for slice_ in region)
            region_power = exponential(sig_power, region_shape)
            psd[region] += region_power
            if mask_mode=='binary':
                mask[region] = 1.0
            elif mask_mode=='snr':
                mask[region] += sig_power/noise_power
            elif mask_mode=='channels':
                region_m=(slice(sig_id, sig_id+1),)+region
                mask[region_m] = 1.0

        return (psd, mask)


    def likelihood(self, S):

        if S is None:
            ll = 0.0
        else:
            S_size = np.prod(S.shape)
            S_mean = np.mean(S)
            ll = S_size * ((S_mean-1)-np.log(S_mean))

        return ll


    def ML_detector(self, psd, thr=0.0):

        shape = np.shape(psd)
        ndims = len(shape)
        self.ll_max = 0.0
        self.S_ML = None

        def sweep_psd(self, start_indices, end_indices):
            if len(start_indices) == ndims:
                slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                subarray = psd[slices]
                ll = self.likelihood(subarray)
                if ll > self.ll_max:
                    self.S_ML = slices
                    self.ll_max = ll
                return

            dim = len(start_indices)
            for start in range(shape[dim]):
                for end in range(start + 1, shape[dim] + 1):
                    sweep_psd(self, start_indices=start_indices + [start], end_indices=end_indices + [end])

        sweep_psd(self, start_indices=[], end_indices=[])
        if self.ll_max<thr:
            self.ll_max = 0.0
            self.S_ML = None

        return(self.S_ML, self.ll_max)


    def ML_detector_efficient(self, psd, thr=0.0):

        shape = np.shape(psd)
        ndims = len(shape)

        self.ll_max = 0.0
        self.S_ML = None

        # compute the cumulative sum over all axises
        psd_cs = self.compute_cumsum(psd)
        if ndims==1:
            psd_cs = np.pad(psd_cs, (1, 0), mode='constant')

            # Compute the matrix of sums for all intervals
            sums = psd_cs[:, None] - psd_cs[None, :]

            # Compute the lengths of intervals
            lens = np.arange(0, len(psd) + 1)

            # Broadcast the lengths to match the shape of sums
            lens = lens[:, None] - lens[None, :]

            # Compute the LLRs
            means = sums/lens
            llrs = lens*((means-1)-np.log(means))
            llrs = np.nan_to_num(llrs, nan=0.0)
            llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)

            self.S_ML = (slice(llrs_max_idx[1], llrs_max_idx[0]),)
            self.ll_max = llrs[llrs_max_idx]
        

        elif ndims==2:
            rows, cols = psd.shape
            psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')

            # Compute the matrix of sums for all intervals
            sums = psd_cs[:,None,:,None] + psd_cs[None,:,None,:] - psd_cs[:,None,None,:] - psd_cs[None,:,:,None]

            # Compute the area of intervals
            row_indices = np.arange(rows+1)[:, None]
            col_indices = np.arange(cols+1)[None, :]
            area_1 = row_indices*col_indices
            area = area_1[:,None,:,None] + area_1[None,:,None,:] - area_1[:,None,None,:] - area_1[None,:,:,None]

            # Compute the LLRs
            means = sums/area
            llrs = area*((means-1)-np.log(means))
            llrs = np.nan_to_num(llrs, nan=0.0)
            llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)

            s1 = [llrs_max_idx[0], llrs_max_idx[1]]
            s2 = [llrs_max_idx[2], llrs_max_idx[3]]
            self.S_ML = (slice(min(s1), max(s1)),slice(min(s2), max(s2)))
            self.ll_max = llrs[llrs_max_idx]

        # elif ndims==2:

        #     rows, cols = psd.shape

        #     for i1 in range(rows):
        #         for j1 in range(cols):
        #             for i2 in range(i1, rows):
        #                 for j2 in range(j1, cols):
        #                     area = (i2 - i1 + 1) * (j2 - j1 + 1)
                            
        #                     sum = psd_cs[i2, j2]
        #                     if i1 > 0:
        #                         sum -= psd_cs[i1 - 1, j2]
        #                     if j1 > 0:
        #                         sum -= psd_cs[i2, j1 - 1]
        #                     if i1 > 0 and j1 > 0:
        #                         sum += psd_cs[i1 - 1, j1 - 1]

        #                     mean = sum / area
        #                     llr = area*((mean-1)-np.log(mean))
                            
        #                     if llr > self.ll_max:
        #                         self.ll_max = llr
        #                         self.S_ML = (i1, j1, i2, j2)

        if self.ll_max<thr:
            self.ll_max = 0.0
            self.S_ML = None

        return(self.S_ML, self.ll_max)
    

    def compute_cumsum(self, X):
        F = X.copy()
        for axis in range(F.ndim):
            F = np.cumsum(F, axis=axis)
        return F


    def sweep_combs_psd(self, psd, S_ML):
        shape = np.shape(psd)
        ndims = len(shape)
        if ndims!=2:
            print("Only dim 2 is supported in this function!")
            return
        psd_cs = self.compute_cumsum(psd)
        psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
        rows, cols = psd.shape

        row_indices = np.arange(rows+1)[:, None]
        col_indices = np.arange(cols+1)[None, :]
        area_1 = row_indices*col_indices

        i1=S_ML[0].start
        j1=S_ML[0].stop
        i2=S_ML[1].start
        j2=S_ML[1].stop
        indices = [0, 1, 2, 3, 4, 5]

        psd_cs_list=[]
        psd_cs_list.append(psd_cs[None,None,:,:])
        psd_cs_list.append(psd_cs[:,:,None,None])
        psd_cs_list.append(psd_cs[:,None,None,:])
        psd_cs_list.append(psd_cs[:,None,:,None])
        psd_cs_list.append(psd_cs[None,:,None,:])
        psd_cs_list.append(psd_cs[None,:,:,None])

        area_list=[]
        area_list.append(area_1[None,None,:,:])
        area_list.append(area_1[:,:,None,None])
        area_list.append(area_1[:,None,None,:])
        area_list.append(area_1[:,None,:,None])
        area_list.append(area_1[None,:,None,:])
        area_list.append(area_1[None,:,:,None])

        combs = itertools.combinations(indices, 4)
        results = []

        for comb in combs:
            # Generate all permutations of the selected 4 indices
            perms = itertools.permutations(comb)
            for perm in perms:

                sums = psd_cs_list[perm[0]] + psd_cs_list[perm[1]] - psd_cs_list[perm[2]] - psd_cs_list[perm[3]]
                area = area_list[perm[0]] + area_list[perm[1]] - area_list[perm[2]] - area_list[perm[3]]
                means = sums/area
                llrs = area*((means-1)-np.log(means))
                llrs = np.nan_to_num(llrs, nan=0.0)
                llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                S_ML = llrs_max_idx
                ll_max = llrs[llrs_max_idx]
                if i1 in S_ML and i2 in S_ML and j1 in S_ML and j2 in S_ML:
                    results.append((perm, S_ML))

        print(results)
        return results


    def get_hyperrectangle_sum(self, F, bounds):
        # bounds is a list of tuples [(a1, b1), (a2, b2), ..., (ad, bd)]
        sum_S = F[tuple(b - 1 for a, b in bounds)]
        for i in range(1, 1 << len(bounds)):
            sub_bounds = []
            sign = -1
            for j in range(len(bounds)):
                if i & (1 << j):
                    sub_bounds.append(bounds[j][0] - 1)
                    sign *= -1
                else:
                    sub_bounds.append(bounds[j][1] - 1)
            if all(b >= 0 for b in sub_bounds):
                sum_S += sign * F[tuple(sub_bounds)]
        return sum_S


    def compute_llrs_ndim(self, X):
        F = self.compute_cumsum(X)
        shape = X.shape
        LLRs = np.zeros(shape + shape)
        
        # Generate all hyper-rectangles
        for bounds in np.ndindex(*([2] * X.ndim)):
            for indices in np.ndindex(*shape):
                if all(b == 0 or b == s for b, s in zip(bounds, indices)):
                    continue
                a_indices = tuple(0 if b == 0 else i for b, i in zip(bounds, indices))
                b_indices = tuple(i + 1 if b == 1 else i for b, i in zip(bounds, indices))
                sub_bounds = list(zip(a_indices, b_indices))
                volume_S = np.prod([b - a for a, b in sub_bounds])
                if volume_S > 0:
                    sum_S = self.get_hyperrectangle_sum(F, sub_bounds)
                    mean_S = sum_S / volume_S
                    idx = tuple(slice(a, b) for a, b in sub_bounds)
                    LLRs[idx] = volume_S * max(mean_S - 1, 0)
        
        return LLRs


    def slice_size(self, slice=None):
        if slice is None:
            size = 0
        else:
            size = 1
            for s in slice:
                size *= (s.stop - s.start)
        return size


    def slice_intersection(self, slice_1, slice_2):
        intersect = []
        if slice_1 is None or slice_2 is None:
            return None
        for s1, s2 in zip(slice_1, slice_2):
            start = max(s1.start, s2.start)
            stop = min(s1.stop, s2.stop)
            if start < stop:
                intersect.append(slice(start, stop))
            else:
                # If the slices do not intersect
                return None
        return tuple(intersect)
    

    def slice_union(self, slice_1, slice_2):
        union = []
        if slice_1 is None:
            return slice_2
        elif slice_2 is None:
            return slice_1
        for s1, s2 in zip(slice_1, slice_2):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            if start < stop:
                union.append(slice(start, stop))
            else:
                # If the slices do not intersect
                return None
        return tuple(union)


    def compute_slices_similarity(self, slice_1, slice_2):
        if slice_1 is None and slice_2 is not None:
            ratio = 0.0
        if slice_2 is None and slice_1 is not None:
            ratio = 0.0
        if slice_1 is None and slice_2 is None:
            ratio = 1.0
        else:
            intersection = self.slice_intersection(slice_1, slice_2)
            union = self.slice_union(slice_1, slice_2)
            intersection_size = self.slice_size(intersection)
            union_size = self.slice_size(union)

            # max_size = max(self.slice_size(slice_1), self.slice_size(slice_2))
            # ratio = intersection_size / max_size
            ratio = intersection_size / union_size

        return ratio


    def find_ML_thr(self, thr_coeff=1.0):
        print("Starting to find the optimal ML threshold...")
        
        ll_list = []
        for i in range(10):
            n_sigs = 0
            psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=np.array([10]), mask_mode='binary')
            (S_ML, ll_max) = self.ML_detector_efficient(psd)
            ll_list.append(ll_max)
        
        ll_mean = np.mean(np.array(ll_list))
        optimal_thr = thr_coeff*ll_mean
        self.ML_thr = optimal_thr
        print("Optimal ML threshold: {}".format(optimal_thr))

        return optimal_thr
    

    def sweep_snrs(self, snrs, n_sigs_min=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None):
        print("Starting to sweep ML detector on SNRs...")
        
        det_rate = {}
        for snr in snrs:
            det_rate[snr] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, SNR: {}'.format(i+1, snr))
                n_sigs = randint(n_sigs_min, n_sigs_max+1)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max)
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=np.array([snr]), mask_mode='binary')
                (S_ML, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr)
                region_gt = regions[0] if len(regions)>0 else None
                det_rate[snr] += self.compute_slices_similarity(S_ML, region_gt)/self.n_simulations

        return det_rate

    
    def sweep_sizes(self, sizes, n_sigs_min=1, n_sigs_max=1, snrs=np.array([10])):
        print("Starting to sweep ML detector on Signal sizes...")
        
        det_rate = {}
        for size in sizes:
            det_rate[size] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, Size: {}'.format(i+1, size))
                n_sigs = randint(n_sigs_min, n_sigs_max+1)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=size, max_size=size)
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=snrs, mask_mode='binary')
                (S_ML, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr)
                region_gt = regions[0] if len(regions)>0 else None
                det_rate[size] += self.compute_slices_similarity(S_ML, region_gt)/self.n_simulations

        return det_rate


    def generate_psd_dataset(self, dataset_path='./data/psd_dataset.npz', n_dataset=1000, shape=(1000,), n_sigs_min=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None, snrs=np.array([10]), mask_mode='binary'): 
        print("Starting to generate PSD dataset with n_dataset={}, shape={}, n_sigs={}-{}, sig_size={}-{}, snrs={}...".format(n_dataset, shape, n_sigs_min, n_sigs_max, sig_size_min, sig_size_max, snrs))
        
        data = []
        masks = []
        for _ in range(n_dataset):
            n_sigs = randint(n_sigs_min, n_sigs_max+1)
            (psd, mask) = self.generate_random_PSD(shape=shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs_max, sig_size_min=sig_size_min, sig_size_max=sig_size_max, noise_power=self.noise_power, snrs=snrs, mask_mode=mask_mode)
            data.append(psd)
            masks.append(mask)
        data = np.array(data)
        masks = np.array(masks)
        np.savez(dataset_path, data=data, masks=masks)

        print(f"Dataset of data shape {data.shape} and mask shape {masks.shape} saved to {dataset_path}")


    def gen_random_str(self, length=10):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(length))


    def plot(self, plot_dic, snrs=None, sizes=None):
        colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'purple']
        plt.figure()
        for i, plot_name in enumerate(plot_dic.keys()):
            if plot_name=='snr_ML':
                x = snrs.copy()
                param_name = 'SNR'
                method = 'Maximum Likelihood'
            elif plot_name=='size_ML':
                x = sizes.copy()
                param_name = 'Interval Size'
                method = 'Maximum Likelihood'
            elif plot_name=='snr_NN':
                x = snrs.copy()
                param_name = 'SNR'
                method = 'U-Net'
            elif plot_name=='size_NN':
                x = sizes.copy()
                param_name = 'Interval Size'
                method = 'U-Net'
            det_err = 1-np.array(list(plot_dic[plot_name].values()))

            plt.semilogx(x, det_err, 'o-', color=colors[i], label=method)
        
        plt.title('Detector Error Rate vs {}'.format(param_name))
        plt.xlabel('{} (Logarithmic)'.format(param_name))
        plt.ylabel('Error Rate')
        plt.legend()
        plt.savefig(self.figs_dir + '{}.pdf'.format(param_name), format='pdf')
        plt.show()



if __name__ == '__main__':
    # ss_det = specsense_detection(params)
    random_string = specsense_detection.gen_random_str(None, length=10)
    print(random_string)

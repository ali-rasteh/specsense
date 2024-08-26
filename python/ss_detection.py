import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, rand, randint, choice, exponential, uniform
from filter_utils import *
import string
import random
import itertools
from scipy.stats import chi2
import torch
import heapq



class specsense_detection(object):
    def __init__(self, params):

        self.shape = params.shape
        self.size_sam_mode = params.size_sam_mode
        self.snr_sam_mode = params.snr_sam_mode
        self.n_simulations = params.n_simulations
        self.noise_power = params.noise_power
        self.ML_thr = params.ML_thr
        self.ML_mode = params.ML_mode
        self.figs_dir = params.figs_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('specsense_detection device: {}'.format(self.device))

        print("Initialized Spectrum Sensing class instance.")


    def plot_MD_vs_SNR(self, snr_min=0.1, snr_max=100.0, n_points=1000, dof_min=1, dof_max=1024, n_dof=11, p_fa=1e-6):

        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_points)
        dofs = np.logspace(np.log10(dof_min), np.log10(dof_max), n_dof).round().astype(int)
        # dofs = np.linspace(dof_min, dof_max, n_dof).round().astype(int)
        seen = set()
        dofs = [x for x in dofs if not (x in seen or seen.add(x))]
        dofs = np.array(dofs)

        plt.figure(figsize=(8, 6))
        for dof in dofs:
            x = chi2.ppf(1-p_fa, dof)
            p_md = chi2.cdf(x/(1+snrs), dof)
            plt.plot(10*np.log10(snrs), p_md, label=f'DoF={dof}')
        plt.title('Probability of Missed Detection vs SNR for Different DoFs')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Probability of Missed Detection')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figs_dir + 'md_vs_snr_dof.pdf', format='pdf')
        plt.show()


    def plot_MD_vs_DoF(self, dof_min=1, dof_max=1024, n_points=1000, snr_min=0.25, snr_max=64, n_snr=9, p_fa=1e-6):

        dofs = np.logspace(np.log10(dof_min), np.log10(dof_max), n_points).round().astype(int)
        # dofs = np.linspace(dof_min, dof_max, n_points).round().astype(int)
        seen = set()
        dofs = [x for x in dofs if not (x in seen or seen.add(x))]
        dofs = np.array(dofs)
        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_snr)

        plt.figure(figsize=(8, 6))
        for snr in snrs:
            x = chi2.ppf(1-p_fa, dofs)
            p_md = chi2.cdf(x/(1+snr), dofs)
            # plt.plot(dofs, np.log(p_md), label='SNR={:0.2f}'.format(snr))
            plt.semilogx(dofs, p_md, label='SNR={:0.2f}'.format(snr))
        plt.title('Probability of Missed Detection vs DoF for Different SNRs')
        plt.xlabel('DoF (Logarithmic)')
        plt.ylabel('Probability of Missed Detection')
        # plt.xlabel('DoF')
        # plt.ylabel('Probability of Missed Detection (Logarithmic)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figs_dir + 'md_vs_dof_snr.pdf', format='pdf')
        plt.show()


    def generate_random_regions(self, shape=(1000,), n_regions=1, min_size=None, max_size=None, size_sam_mode='log'):
        regions = []
        ndims = len(shape)
        for _ in range(n_regions):
            region_slices = []
            for d, dim in enumerate(shape):
                if min_size is not None and max_size is not None:
                    s1 = min_size[d]
                    s2 = max_size[d] + 1
                else:
                    s1 = 1
                    s2 = min(101, (dim+1)//2+1)
                if size_sam_mode=='lin':
                    size = randint(s1, s2)
                elif size_sam_mode=='log':
                    margin=1e-9
                    size = uniform(np.log10(s1), np.log10(s2-margin))
                    size = int(10 ** size)
                start = randint(0, dim-size+1)
                size = min(size, dim-start)
                region_slices.append(slice(start, start + size))
            regions.append(tuple(region_slices))

        return regions


    def generate_random_PSD(self, shape=(1000,), sig_regions=None, n_sigs=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None, noise_power=1, snr_range=np.array([10,10]), size_sam_mode='log', snr_sam_mode='log', mask_mode='binary'):

        sig_power_range = noise_power * snr_range.astype(float)
        psd = exponential(noise_power, shape)
        if mask_mode=='binary' or mask_mode=='snr':
            mask = np.zeros(shape, dtype=float)
        elif mask_mode=='channels':
            mask = np.zeros((n_sigs_max,)+shape, dtype=float)

        if sig_regions is None:
            regions = self.generate_random_regions(shape=shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max, size_sam_mode=size_sam_mode)
        else:
            regions = sig_regions

        for sig_id, region in enumerate(regions):
            if snr_sam_mode=='lin':
                # sig_power = choice(sig_powers)
                sig_power = uniform(sig_power_range[0], sig_power_range[1])
            elif snr_sam_mode=='log':
                sig_power = uniform(np.log10(sig_power_range[0]), np.log10(sig_power_range[1]))
                sig_power = 10**sig_power
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
        ll_max = 0.0
        S_ML = None

        def sweep_psd(self, start_indices, end_indices):
            if len(start_indices) == ndims:
                slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                subarray = psd[slices]
                ll = self.likelihood(subarray)
                if ll > ll_max:
                    S_ML = slices
                    ll_max = ll
                return

            dim = len(start_indices)
            for start in range(shape[dim]):
                for end in range(start + 1, shape[dim] + 1):
                    sweep_psd(self, start_indices=start_indices + [start], end_indices=end_indices + [end])

        sweep_psd(self, start_indices=[], end_indices=[])
        if ll_max<thr:
            ll_max = 0.0
            S_ML = None

        return(S_ML, ll_max)


    def ML_detector_efficient(self, psd, thr=0.0, mode='np'):

        ll_max = 0.0
        S_ML = None

        if mode=='np':
            shape = np.shape(psd)
        elif mode=='torch':
            psd = torch.tensor(psd, dtype=torch.float64)
            psd = psd.to(self.device)
            shape = psd.shape
        ndims = len(shape)
        
        # compute the cumulative sum over all axises
        psd_cs = self.compute_cumsum(psd, mode=mode)
        if ndims==1:
            if mode=='np':
                psd_cs = np.pad(psd_cs, (1, 0), mode='constant')
            elif mode=='torch':
                psd_cs = torch.nn.functional.pad(psd_cs, (1, 0), mode='constant', value=0)

            # Compute the matrix of sums for all intervals
            sums = psd_cs[:, None] - psd_cs[None, :]

            # Compute the lengths of intervals
            if mode=='np':
                lens = np.arange(0, len(psd) + 1)
            elif mode=='torch':
                lens = torch.arange(0, len(psd) + 1, device=self.device)

            # Broadcast the lengths to match the shape of sums
            lens = lens[:, None] - lens[None, :]

            # Compute the LLRs
            means = sums/lens
            if mode=='np':
                llrs = lens*((means-1)-np.log(means))
                llrs = np.nan_to_num(llrs, nan=0.0)
                llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                S_ML = (slice(llrs_max_idx[1], llrs_max_idx[0]),)
                ll_max = llrs[llrs_max_idx]
            elif mode=='torch':
                llrs = lens*((means-1)-torch.log(means))
                llrs = torch.nan_to_num(llrs, nan=0.0)
                try:
                    llrs_max_idx = torch.unravel_index(torch.argmax(llrs), llrs.shape)
                except:
                    llrs_max_idx = np.unravel_index(torch.argmax(llrs).cpu().numpy(), llrs.shape)
                S_ML = (slice(llrs_max_idx[1].item(), llrs_max_idx[0].item()),)
                ll_max = llrs[llrs_max_idx].item()

        elif ndims==2:
            rows, cols = psd.shape
            if mode=='np':
                psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
            elif mode=='torch':
                psd_cs = torch.nn.functional.pad(psd_cs, (1, 0, 1, 0), mode='constant', value=0)
           
            # Compute the matrix of sums for all intervals
            sums = psd_cs[:,None,:,None] + psd_cs[None,:,None,:] - psd_cs[:,None,None,:] - psd_cs[None,:,:,None]

            # Compute the area of intervals
            if mode=='np':
                row_indices = np.arange(rows+1)[:, None]
                col_indices = np.arange(cols+1)[None, :]
            elif mode=='torch':
                row_indices = torch.arange(rows + 1, device=self.device)[:, None]
                col_indices = torch.arange(cols + 1, device=self.device)[None, :]
            area_1 = row_indices*col_indices
            area = area_1[:,None,:,None] + area_1[None,:,None,:] - area_1[:,None,None,:] - area_1[None,:,:,None]

            # Compute the LLRs
            means = sums/area
            if mode=='np':
                llrs = area*((means-1)-np.log(means))
                llrs = np.nan_to_num(llrs, nan=0.0)
                llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                s1 = [llrs_max_idx[0], llrs_max_idx[1]]
                s2 = [llrs_max_idx[2], llrs_max_idx[3]]
                S_ML = (slice(min(s1), max(s1)),slice(min(s2), max(s2)))
                ll_max = llrs[llrs_max_idx]
            elif mode=='torch':
                llrs = area * ((means - 1) - torch.log(means))
                llrs = torch.nan_to_num(llrs, nan=0.0)
                try:
                    llrs_max_idx = torch.unravel_index(torch.argmax(llrs), llrs.shape)
                except:
                    llrs_max_idx = np.unravel_index(torch.argmax(llrs).cpu().numpy(), llrs.shape)
                s1 = [llrs_max_idx[0].item(), llrs_max_idx[1].item()]
                s2 = [llrs_max_idx[2].item(), llrs_max_idx[3].item()]
                S_ML = (slice(min(s1), max(s1)), slice(min(s2), max(s2)))
                ll_max = llrs[llrs_max_idx].item()

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
                            
        #                     if llr > ll_max:
        #                         ll_max = llr
        #                         S_ML = (i1, j1, i2, j2)

        if ll_max<thr:
            ll_max = 0.0
            S_ML = None

        return (S_ML, ll_max)
    

    def ML_detector_binary_search(self, psd, adj_search=1, n_largest=3, thr=0.0, mode='np'):
        ll_max = 0.0
        S_ML = None
        n_channels_max = 1
        ll_list=[]

        shape = psd.shape
        ndims = len(shape)
        if ndims==1:
            n_fft = shape[0]
            n_stage = int(np.round(np.log2(shape[0]))) + 1
            for i in range(n_stage):
                n_channels = 2 ** (i)
                n_features = int(n_fft/n_channels)
                lls=[]
                for j in range(n_features):
                    lls.append(self.likelihood(psd[j*n_channels:(j+1)*n_channels]))
                if np.max(lls)>ll_max:
                    ll_max = np.max(lls)
                    S_ML = (slice(np.argmax(lls)*n_channels, (np.argmax(lls)+1)*n_channels),)
                    n_channels_max = n_channels

                largest_lls = heapq.nlargest(n_largest, lls)
                ll_list = ll_list + [((slice(idx*n_channels, (idx+1)*n_channels),), ll, n_channels) for idx, ll in enumerate(lls) if ll in largest_lls]
                
            S_ML_list = [item[0] for item in ll_list]
            ll_max_list = [item[1] for item in ll_list]
            n_channels_list = [item[2] for item in ll_list]
            largest_lls = heapq.nlargest(n_largest, ll_max_list)
            ll_list = [(S_ML_list[idx],ll_max_list[idx],n_channels_list[idx]) for idx, ll in enumerate(ll_max_list) if ll in largest_lls]
            

            for (S_ML_c, ll_max_c, n_channels) in ll_list:
                start = max(S_ML_c[0].start-adj_search*n_channels, 0)
                stop = min(S_ML_c[0].stop+adj_search*n_channels, n_fft)
                (S_ML_m, ll_max_m) = self.ML_detector_efficient(psd=psd[start:stop], thr=thr, mode=mode)
                if (S_ML_m is not None) and ll_max_m>ll_max:
                    S_ML = (slice(start + S_ML_m[0].start, start + S_ML_m[0].stop),)
                    ll_max = ll_max_m

        if ll_max<thr:
            ll_max = 0.0
            S_ML = None
        return(S_ML, ll_max)



    def compute_cumsum(self, X, mode='np'):
        if mode=='np':
            F = X.copy()
            for axis in range(F.ndim):
                F = np.cumsum(F, axis=axis)
        elif mode=='torch':
            F = X.clone()
            for axis in range(F.dim()):
                F = torch.cumsum(F, dim=axis)
        
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
            (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([10,10]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')

            (S_ML, ll_max) = self.ML_detector_efficient(psd, mode=self.ML_mode)
            ll_list.append(ll_max)
        
        ll_mean = np.mean(np.array(ll_list))
        optimal_thr = thr_coeff*ll_mean
        self.ML_thr = optimal_thr
        print("Optimal ML threshold: {}".format(optimal_thr))

        return optimal_thr
    

    def sweep_snrs(self, snrs, n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, sig_size_min=None, sig_size_max=None):
        print("Starting to sweep ML detector on SNRs...")
        
        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        det_rate = {}
        cnt = 0
        for snr in snrs:
            det_rate[snr] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, SNR: {:0.3f}'.format(i+1, snr))
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                # n_sigs = randint(n_sigs_min, n_sigs_max+1)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([snr,snr]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                (S_ML, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode)
                (S_ML_1, ll_max_1) = self.ML_detector_binary_search(psd, adj_search=1, thr=self.ML_thr, mode=self.ML_mode)
                if S_ML_1 != S_ML or np.round(ll_max,3)!=np.round(ll_max_1,3):
                    print((S_ML_1, S_ML))
                    print((ll_max_1, ll_max))
                    cnt += 1
                    # raise ValueError("Binary search ML detector failed!")
                region_gt = regions[0] if len(regions)>0 else None
                det_rate[snr] += self.compute_slices_similarity(S_ML, region_gt)/self.n_simulations

        print("Binary search ML detector failed in {} cases!".format(cnt))
        print("Binary search ML detector failed in {} percent!".format(cnt/(self.n_simulations*len(snrs))*100))
        return det_rate

    
    def sweep_sizes(self, sizes, n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, snr_range=np.array([10,10])):
        print("Starting to sweep ML detector on Signal sizes...")
        
        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        det_rate = {}
        for size in sizes:
            det_rate[size] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, Size: {}'.format(i+1, size))
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                # n_sigs = randint(n_sigs_min, n_sigs_max+1)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=size, max_size=size, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=snr_range, size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                (S_ML, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode)
                region_gt = regions[0] if len(regions)>0 else None
                det_rate[size] += self.compute_slices_similarity(S_ML, region_gt)/self.n_simulations

        return det_rate


    def generate_psd_dataset(self, dataset_path='./data/psd_dataset.npz', n_dataset=1000, shape=(1000,), n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, sig_size_min=None, sig_size_max=None, snr_range=np.array([10,10]), mask_mode='binary'): 
        print("Starting to generate PSD dataset with n_dataset={}, shape={}, n_sigs={}-{}, n_sigs_p_dist:{}, sig_size={}-{}, snrs={:0.3f}-{:0.3f}...".format(n_dataset, shape, n_sigs_min, n_sigs_max, n_sigs_p_dist, sig_size_min, sig_size_max, snr_range[0], snr_range[1]))
        
        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        data = []
        masks = []
        for _ in range(n_dataset):
            n_sigs = np.random.choice(n_sigs_list, p=n_sigs_p_dist)
            # n_sigs = randint(n_sigs_min, n_sigs_max+1)
            (psd, mask) = self.generate_random_PSD(shape=shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs_max, sig_size_min=sig_size_min, sig_size_max=sig_size_max, noise_power=self.noise_power, snr_range=snr_range, size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode=mask_mode)
            data.append(psd)
            masks.append(mask)
        data = np.array(data)
        masks = np.array(masks)
        np.savez(dataset_path, data=data, masks=masks)
        
        print(f"Dataset of data shape {data.shape} and mask shape {masks.shape} saved to {dataset_path}")


    def gen_random_str(self, length=6):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(length))


    def plot(self, plot_dic):
        colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'purple']
        plt.figure()
        for i, plot_name in enumerate(plot_dic.keys()):
            x = [float(i) for i in plot_dic[plot_name].keys()]

            if 'snr' in plot_name:
                param_name = 'SNR'
                file_name = 'ss_sw_snr'
            elif 'size' in plot_name:
                param_name = 'Interval Size'
                file_name = 'ss_sw_size'

            if 'ML' in plot_name:
                method = 'Maximum Likelihood'
            elif 'NN' in plot_name:
                method = 'U-Net'

            det_err = 1-np.array(list(plot_dic[plot_name].values()))
            plt.semilogx(x, det_err, 'o-', color=colors[i], label=method)
        
        plt.title('Detector Error Rate vs {}'.format(param_name))
        plt.xlabel('{} (Logarithmic)'.format(param_name))
        plt.ylabel('Error Rate')
        plt.legend()
        plt.savefig(self.figs_dir + '{}.pdf'.format(file_name), format='pdf')
        plt.show()



if __name__ == '__main__':
    # ss_det = specsense_detection(params)
    random_string = specsense_detection.gen_random_str(None, length=10)
    print(random_string)

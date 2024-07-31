import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand, randint, choice, exponential
from filter_utils import *
import time
import itertools



class specsense_detection(object):
    def __init__(self, params):

        self.shape = params.shape
        self.n_simulations = params.n_simulations
        self.noise_power = params.noise_power

        print("Initialized Spectrum Sensing class instance.")


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

        S_size = np.prod(S.shape)
        S_mean = np.mean(S)
        ll = S_size * ((S_mean-1)-np.log(S_mean))

        return ll


    def ML_detector(self, psd, thr=0.5):

        shape = np.shape(psd)
        ndims = len(shape)
        self.ll_max = 0
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
        return(self.S_ML, self.ll_max)


    def ML_detector_efficient(self, psd, thr=0.5):
        shape = np.shape(psd)
        ndims = len(shape)

        self.ll_max = 0
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
            # lens = lens + np.eye(len(psd)+1, dtype=float)

            # Compute the LLRs
            # J = lens * np.maximum(Xs - 1, 0)
            means = sums/lens
            llrs = lens*((means-1)-np.log(means))
            llrs = np.nan_to_num(llrs, nan=0.0)
            llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)

            self.S_ML = (slice(llrs_max_idx[1], llrs_max_idx[0]),)
            self.ll_max = llrs[llrs_max_idx]
        

        elif ndims==2:
            rows, cols = psd.shape
            # print(psd)
            # print(psd_cs)
            psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
            # print(psd_cs)
            # Compute the matrix of sums for all intervals
            # sums = psd_cs[:,:,None,None] - psd_cs[None,None,:,:]
            sums = psd_cs[:,None,:,None] + psd_cs[None,:,None,:] - psd_cs[:,None,None,:] - psd_cs[None,:,:,None]



            # sums = (psd_cs[1:, 1:, None, None] - psd_cs[:-1, 1:, None, None]
            # - psd_cs[1:, :-1, None, None] + psd_cs[:-1, :-1, None, None])

            # Compute the lengths of intervals
            row_indices = np.arange(rows+1)[:, None]
            col_indices = np.arange(cols+1)[None, :]
            area_1 = row_indices*col_indices
            # print(area_1)
            # area = area_1[:,:,None,None] - area_1[None,None,:,:]
            area = area_1[:,None,:,None] + area_1[None,:,None,:] - area_1[:,None,None,:] - area_1[None,:,:,None]


            # i1=S_ML_t[0].start
            # j1=S_ML_t[0].stop
            # i2=S_ML_t[1].start
            # j2=S_ML_t[1].stop
            # indices = [0, 1, 2, 3, 4, 5]
            # psd_cs_list=[]
            # psd_cs_list.append(psd_cs[None,None,:,:])
            # psd_cs_list.append(psd_cs[:,:,None,None])
            # psd_cs_list.append(psd_cs[:,None,None,:])
            # psd_cs_list.append(psd_cs[:,None,:,None])
            # psd_cs_list.append(psd_cs[None,:,None,:])
            # psd_cs_list.append(psd_cs[None,:,:,None])
            # area_list=[]
            # area_list.append(area_1[None,None,:,:])
            # area_list.append(area_1[:,:,None,None])
            # area_list.append(area_1[:,None,None,:])
            # area_list.append(area_1[:,None,:,None])
            # area_list.append(area_1[None,:,None,:])
            # area_list.append(area_1[None,:,:,None])
            # combs = itertools.combinations(indices, 4)
            # results = []

            # for comb in combs:
            #     # Generate all permutations of the selected 4 indices
            #     perms = itertools.permutations(comb)
            #     for perm in perms:

            #         sums = psd_cs_list[perm[0]] + psd_cs_list[perm[1]] - psd_cs_list[perm[2]] - psd_cs_list[perm[3]]
            #         area = area_list[perm[0]] + area_list[perm[1]] - area_list[perm[2]] - area_list[perm[3]]
            #         means = sums/area
            #         llrs = area*((means-1)-np.log(means))
            #         llrs = np.nan_to_num(llrs, nan=0.0)
            #         llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
            #         S_ML = llrs_max_idx
            #         ll_max = llrs[llrs_max_idx]
            #         if i1 in S_ML and i2 in S_ML and j1 in S_ML and j2 in S_ML:
            #             results.append((perm, S_ML))

            # print(results)
            # raise ValueError("terminate")
            # return results



            # area = (area_1[1:, 1:, None, None] - area_1[:-1, 1:, None, None]
            # - area_1[1:, :-1, None, None] + area_1[:-1, :-1, None, None])

            # i, j = np.indices((rows + 1, cols + 1))
            # area = (i[1:, 1:, None, None] - i[:-1, 1:, None, None]) * (j[1:, 1:, None, None] - j[1:, :-1, None, None])



            # i_indices = np.arange(1, rows + 1)[:, None]
            # j_indices = np.arange(1, cols + 1)[None, :]
            # sums = (psd_cs[i_indices, j_indices] -
            # psd_cs[i_indices - 1, j_indices] -
            # psd_cs[i_indices, j_indices - 1] +
            # psd_cs[i_indices - 1, j_indices - 1])
            # area = (i_indices - i_indices.T + 1) * (j_indices - j_indices.T + 1)


            # Compute the LLRs
            means = sums/area
            llrs = area*((means-1)-np.log(means))
            llrs = np.nan_to_num(llrs, nan=0.0)
            llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)

            s1 = [llrs_max_idx[0], llrs_max_idx[1]]
            s2 = [llrs_max_idx[2], llrs_max_idx[3]]
            self.S_ML = (slice(min(s1), max(s1)),slice(min(s2), max(s2)))
            # self.S_ML = llrs_max_idx
            self.ll_max = llrs[llrs_max_idx]

            # raise ValueError("terminate")

        
        elif ndims==3:
            print(psd)
            # Initialize the dimensions
            rows, cols = psd.shape
            # print(rows, cols)

            # Create the arrays for indexing
            row_indices = np.arange(rows)[:, None]
            col_indices = np.arange(cols)[None, :]
            # print(row_indices)
            # print(col_indices)

            # Compute all submatrix sums using broadcasting
            print(psd_cs)
            psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
            print(psd_cs)
            # sums = (psd_cs[row_indices + 1, col_indices + 1]
            #     - psd_cs[row_indices + 1, col_indices]
            #     - psd_cs[row_indices, col_indices + 1]
            #     + psd_cs[row_indices, col_indices])
            # print(sums)

            # Generate all possible top-left and bottom-right corners of submatrices
            # a, b = np.tril_indices(rows)
            # c, d = np.tril_indices(cols)
            a, b = np.triu_indices(rows)
            c, d = np.triu_indices(cols)
            print(a,b)
            print(c,d)

            areas = (b - a + 1).reshape(-1, 1, 1) * (d - c + 1).reshape(1, -1, 1)
            # print(areas)

            print(psd_cs[a, c])
            print(psd_cs[b + 1, d + 1]
            - psd_cs[a, d + 1]
            - psd_cs[b + 1, c]
            + psd_cs[a, c])
            submatrix_sums = (
            psd_cs[b + 1, d + 1]
            - psd_cs[a, d + 1][:, :, None]
            - psd_cs[b + 1, c][:, None, :]
            + psd_cs[a, c][:, :, None]
            )
            # submatrix_sums = sums[a, b, :, :][:, :, c, d]
            print(submatrix_sums)

            # Compute the lengths (area) of the submatrices
            lens_row = np.arange(1, rows + 1)[:, None]
            lens_col = np.arange(1, cols + 1)[None, :]
            lens = lens_row * lens_col
            # print(lens)

            # Compute the means for each submatrix
            means = sums / lens
            # print(means)
            llrs = lens*((means-1)-np.log(means))
            # print(llrs)
            llrs = np.nan_to_num(llrs, nan=0.0)
            # print(llrs)
            llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
            print(llrs_max_idx)
            print(llrs[llrs_max_idx])

            self.S_ML = (slice(llrs_max_idx[1], llrs_max_idx[0]),)
            self.ll_max = llrs[llrs_max_idx]

            raise ValueError("terminate")


        return(self.S_ML, self.ll_max)
    
    
    def compute_interval(self, matrix):
        # Compute the cumulative sum over the 2 axes
        cumsum_matrix = np.cumsum(np.cumsum(matrix, axis=0), axis=1)

        max_value = float('-inf')
        max_coords = (0, 0, 0, 0)  # (i1, j1, i2, j2)

        rows, cols = matrix.shape

        for i1 in range(rows):
            for j1 in range(cols):
                for i2 in range(i1, rows):
                    for j2 in range(j1, cols):
                        area = (i2 - i1 + 1) * (j2 - j1 + 1)
                        
                        total_sum = cumsum_matrix[i2, j2]
                        if i1 > 0:
                            total_sum -= cumsum_matrix[i1 - 1, j2]
                        if j1 > 0:
                            total_sum -= cumsum_matrix[i2, j1 - 1]
                        if i1 > 0 and j1 > 0:
                            total_sum += cumsum_matrix[i1 - 1, j1 - 1]

                        mean = total_sum / area
                        value = area * ((mean - 1) - np.log(mean))
                        
                        if value > max_value:
                            max_value = value
                            max_coords = (i1, j1, i2, j2)

        return max_coords, max_value


    def compute_cumsum(self, X):
        F = X.copy()
        for axis in range(F.ndim):
            F = np.cumsum(F, axis=axis)
        return F


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

        intersection = self.slice_intersection(slice_1, slice_2)
        union = self.slice_union(slice_1, slice_2)
        intersection_size = self.slice_size(intersection)
        union_size = self.slice_size(union)

        # max_size = max(self.slice_size(slice_1), self.slice_size(slice_2))
        # ratio = intersection_size / max_size
        ratio = intersection_size / union_size

        return ratio


    def sweep_snrs(self, snrs, n_sigs_min=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None):
        print("Starting to sweep ML detector on SNRs...")
        
        n_sigs = randint(n_sigs_min, n_sigs_max+1)
        det_rate_snrs = {}
        for snr in snrs:
            det_rate_snrs[snr] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, SNR: {}'.format(i+1, snr))
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max)
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=np.array([snr]), mask_mode='binary')
                (S_ML_1, ll_max) = self.ML_detector(psd)
                (S_ML, ll_max) = self.ML_detector_efficient(psd)
                print(S_ML)
                if S_ML != S_ML_1:
                    print(S_ML)
                    print(S_ML_1)
                    raise ValueError("Not equal!!!")
                det_rate_snrs[snr] += self.compute_slices_similarity(S_ML, regions[0])/self.n_simulations
        
        plt.figure()
        plt.semilogx(snrs, 1-np.array(list(det_rate_snrs.values())), 'o-')
        plt.title('ML detector error rate vs SNR')
        plt.xlabel('SNR (Logarithmic)')
        plt.ylabel('Error rate')
        plt.show()

        return det_rate_snrs

    
    def sweep_sizes(self, sizes, n_sigs_min=1, n_sigs_max=1, snrs=np.array([10])):
        print("Starting to sweep ML detector on Signal sizes...")
        
        n_sigs = randint(n_sigs_min, n_sigs_max+1)
        det_rate_sizes = {}
        for size in sizes:
            det_rate_sizes[size] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, Size: {}'.format(i+1, size))
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=size, max_size=size)
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=snrs, mask_mode='binary')
                (S_ML_1, ll_max) = self.ML_detector(psd)
                (S_ML, ll_max) = self.ML_detector_efficient(psd)
                if S_ML != S_ML_1:
                    print(S_ML)
                    print(S_ML_1)
                    raise ValueError("Not equal!!!")
                det_rate_sizes[size] += self.compute_slices_similarity(S_ML, regions[0])/self.n_simulations
        
        plt.figure()
        plt.semilogx(sizes, 1-np.array(list(det_rate_sizes.values())), 'o-')
        plt.title('ML detector error rate vs interval size')
        plt.xlabel('Interval size (Logarithmic)')
        plt.ylabel('Error rate')
        plt.show()

        return det_rate_sizes


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



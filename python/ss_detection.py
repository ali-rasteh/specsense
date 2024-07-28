import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand, randint, choice, exponential
from filter_utils import *



class specsense_detection(object):
    def __init__(self, params):

        self.dataset_path = params.dataset_path
        self.n_samples = params.n_samples
        self.n_fft = params.n_fft
        self.shape = params.shape
        self.n_sigs_min = params.n_sigs_min
        self.n_sigs_max = params.n_sigs_max
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
                    size = randint(1, min(101, dim//2))
                start = randint(0, dim-size+1)
                size = min(size, dim-start)
                region_slices.append(slice(start, start + size))
            regions.append(tuple(region_slices))
        return regions


    def generate_random_PSD(self, shape=(1000,), sig_regions=None, n_sigs=1, sig_size_min=None, sig_size_max=None, noise_power=1, snrs=np.array([10])):

        sig_powers = noise_power * snrs
        psd = exponential(noise_power, shape)
        mask = np.zeros(shape, dtype=float)

        if sig_regions is None:
            regions = self.generate_random_regions(shape=shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max)
        else:
            regions = sig_regions

        for region in regions:
            sig_power = choice(sig_powers)
            region_shape = tuple(slice_.stop - slice_.start for slice_ in region)
            region_power = exponential(sig_power, region_shape)
            psd[region] += region_power
            mask[region] = 1.0

        return (psd, mask)


    def likelihood(self, S):

        S_size = np.prod(S.shape)
        S_mean = np.mean(S)
        ll = S_size * ((S_mean-1)-np.log(S_mean))

        return ll


    def ML_detector(self, psd):

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


    def sweep_snrs(self, snrs, n_sigs=1, sig_size_min=None, sig_size_max=None):
        print("Starting to sweep ML detector on SNRs...")
        
        det_rate_snrs = {}
        regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max)
        for snr in snrs:
            det_rate_snrs[snr] = 0.0
            for i in range(self.n_simulations):
                print('Simulation #: {}, SNR: {}'.format(i+1, snr))
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=np.array([snr]))
                (S_ML, ll_max) = self.ML_detector(psd)
                det_rate_snrs[snr] += self.compute_slices_similarity(S_ML, regions[0])/self.n_simulations
        
        plt.figure()
        plt.semilogx(snrs, 1-np.array(list(det_rate_snrs.values())), 'o-')
        plt.title('ML detector error rate vs SNR')
        plt.xlabel('SNR (Logarithmic)')
        plt.ylabel('Error rate')
        plt.show()

        return det_rate_snrs

    
    def sweep_sizes(self, sizes, n_sigs=1, snrs=np.array([10])):
        print("Starting to sweep ML detector on Signal sizes...")
        
        det_rate_sizes = {}
        for size in sizes:
            det_rate_sizes[size] = 0.0
            regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=(size,), max_size=(size,))
            for i in range(self.n_simulations):
                print('Simulation #: {}, Size: {}'.format(i+1, size))
                psd, mask = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snrs=snrs)
                (S_ML, ll_max) = self.ML_detector(psd)
                det_rate_sizes[size] += self.compute_slices_similarity(S_ML, regions[0])/self.n_simulations
        
        plt.figure()
        plt.semilogx(sizes, 1-np.array(list(det_rate_sizes.values())), 'o-')
        plt.title('ML detector error rate vs interval size')
        plt.xlabel('Interval size')
        plt.ylabel('Error rate')
        plt.show()

        return det_rate_sizes


    def generate_psd_dataset(self, shape=(1000, 1000,), n_sigs_min=1, n_sigs_max=1, sig_size_min=None, sig_size_max=None, snrs=np.array([10])): 
        print("Starting to generate PSD dataset with shape={}, n_sigs={}-{}, sig_size={}-{}, snrs={}...".format(shape, n_sigs_min, n_sigs_max, sig_size_min, sig_size_max, snrs))
        
        data = []
        masks = []
        for _ in range(shape[0]):
            n_sigs = randint(n_sigs_min, n_sigs_max+1)
            (psd, mask) = self.generate_random_PSD(shape=shape[1:], sig_regions=None, n_sigs=n_sigs, sig_size_min=sig_size_min, sig_size_max=sig_size_max, noise_power=self.noise_power, snrs=snrs)
            data.append(psd)
            masks.append(mask)
        data = np.array(data)
        masks = np.array(masks)
        np.savez(self.dataset_path, data=data, masks=masks)

        print(f"Dataset of shape {data.shape} saved to {self.dataset_path}")
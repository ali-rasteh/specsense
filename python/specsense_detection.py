import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand, randint, choice, exponential
from filter_utils import *



class specsense_detection(object):
    def __init__(self, n_samples=1024, n_fft=1024):

        self.n_samples = n_samples
        self.n_fft = n_fft

    def generate_random_regions(self, shape=(1000,), n_regions=1, max_size=None, fixed_size=False):
        regions = []
        ndims = len(shape)
        for _ in range(n_regions):
            region_slices = []
            for d, dim in enumerate(shape):
                if fixed_size and max_size is not None:
                    start = randint(0, dim-max_size[d]+1)
                    size = max_size[d]
                else:
                    start = randint(0, dim)
                    if max_size is not None:
                        size = randint(1, max_size[d] + 1)
                    else:
                        size = randint(1, dim//2)
                size = min(size, dim-start)
                region_slices.append(slice(start, start + size))
            regions.append(tuple(region_slices))
        return regions

    def generate_random_PSD(self, shape=(1000,), sig_regions=None, n_regions=1, noise_power=1, snrs=np.array([10])):

        sig_powers = noise_power * snrs
        psd = exponential(noise_power, shape)

        if sig_regions is None:
            regions = self.generate_random_regions(shape, n_regions)
        else:
            regions = sig_regions

        for region in regions:
            sig_power = choice(sig_powers)
            region_shape = tuple(slice_.stop - slice_.start for slice_ in region)
            region_power = exponential(sig_power, region_shape)
            psd[region] += region_power

        return psd


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


    def compute_slices_similarity(self, slice_1, slice_2):

        intersection = self.slice_intersection(slice_1, slice_2)
        intersection_size = self.slice_size(intersection)

        max_size = max(self.slice_size(slice_1), self.slice_size(slice_2))
        ratio = intersection_size / max_size

        return 1-ratio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand
from specsense_detection import *
import argparse
import os
import sys



n_fft = 1024
n_samples = 1024
n_sigs = 1
snr_min = 0.1
snr_max = 100
region_size_min = 1
region_size_max = 512
n_simulations = 10
sweep_steps = 10


# snrs = np.linspace(start=snr_min, stop=snr_max, num=sweep_steps)
snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), sweep_steps)
sizes = np.logspace(np.log10(region_size_min), np.log10(region_size_max), sweep_steps)
sizes = sizes.astype(int)
ss_det = specsense_detection(n_samples=n_samples, n_fft=n_fft)
det_rate_snrs = {}
det_rate_sizes = {}



regions = ss_det.generate_random_regions(shape=(n_fft,), n_regions=n_sigs, max_size=(20,), fixed_size=True)
for snr in snrs:
    det_rate_snrs[snr] = 0.0
    for i in range(n_simulations):
        print(i, snr)
        # print(regions)
        psd = ss_det.generate_random_PSD(shape=(n_fft,), sig_regions=regions, n_regions=n_sigs, noise_power=1, snrs=np.array([snr]))
        (S_ML, ll_max) = ss_det.ML_detector(psd)
        # print(S_ML)
        det_rate_snrs[snr] += ss_det.compute_slices_similarity(S_ML, regions[0])/n_simulations


for size in sizes:
    det_rate_sizes[size] = 0.0
    regions = ss_det.generate_random_regions(shape=(n_fft,), n_regions=n_sigs, max_size=(size,), fixed_size=True)
    for i in range(n_simulations):
        print(i, size)
        # print(regions)
        psd = ss_det.generate_random_PSD(shape=(n_fft,), sig_regions=regions, n_regions=n_sigs, noise_power=1, snrs=np.array([10]))
        (S_ML, ll_max) = ss_det.ML_detector(psd)
        # print(S_ML)
        det_rate_sizes[size] += ss_det.compute_slices_similarity(S_ML, regions[0])/n_simulations


print(det_rate_snrs)
print(det_rate_sizes)


plt.figure()
plt.semilogx(snrs, 1-np.array(list(det_rate_snrs.values())), 'o-')
plt.title('ML detector error rate vs SNR')
plt.xlabel('SNR (Logarithmic)')
plt.ylabel('Error rate')
plt.show()

plt.figure()
plt.semilogx(sizes, 1-np.array(list(det_rate_sizes.values())), 'o-')
plt.title('ML detector error rate vs interval size')
plt.xlabel('Interval size')
plt.ylabel('Error rate')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand
from fir_separate import *
import argparse
import os
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=float, default=200e6, help="sampling frequency")
    parser.add_argument("--n_samples", type=float, default=2 ** 13, help="number of samples")
    parser.add_argument("--sharp_bw", type=float, default=10e6, help="bandwidth of sharp basis filters")
    parser.add_argument("--base_order_pos", type=float, default=64, help="positive order of smooth basis filters")
    parser.add_argument("--base_order_neg", type=float, default=0, help="negative order of smooth basis filters")
    parser.add_argument("--n_stage", type=float, default=1, help="number of stages of up/down sampling on smooth basis filters")
    parser.add_argument("--us_rate", type=float, default=2, help="upsampling rate")
    parser.add_argument("--ds_rate", type=float, default=2, help="downsampling rate")
    parser.add_argument("--fil_bank_mode", type=float, default=2,
                        help="mode of filtering bank, 1 for whole-span coverage and 2 for TX signal coverage")
    parser.add_argument("--fil_mode", type=float, default=3,
                        help="mode of filtering, 1: use sharp filter bank, 2: use fir_us filters, 3: use fir_ds_us filters")
    parser.add_argument("--N_sig", type=float, default=8, help="number of TX signals")
    parser.add_argument("--N_r", type=float, default=8, help="number of RX signals (# of antennas)")
    parser.add_argument("--snr", type=float, default=10, help="SNR of the received signal in dB")
    parser.add_argument("--ridge_coeff", type=float, default=1, help="Ridge regression coefficient")
    parser.add_argument("--sig_sel_id", type=float, default=0, help="selected TX signal id for plots, etc")
    parser.add_argument("--rx_sel_id", type=float, default=0, help="selected RX signal id for plots, etc")
    parser.add_argument("--rand_params", help="mode of filtering", action="store_true", default=False)
    parser.add_argument("--plot_level", type=float, default=0, help="level of plotting outputs")
    parser.add_argument("--figs_dir", type=str, default='./figs/', help="directory to save figures")
    parser.add_argument("--sig_noise", help="Add noise to the signals?", action="store_true", default=False)
    args = parser.parse_args()

    # Constants
    fs = args.fs
    n_samples = args.n_samples
    sharp_bw = args.sharp_bw
    base_order_pos = args.base_order_pos
    base_order_neg = args.base_order_neg
    n_stage = args.n_stage
    us_rate = args.us_rate
    ds_rate = args.ds_rate
    fil_bank_mode = args.fil_bank_mode  # 1 for whole-span coverage and 2 for TX signal coverage
    fil_mode = args.fil_mode  # 1: use sharp filter bank, 2: use fir_us filters, 3: use fir_ds_us filters
    N_sig = args.N_sig
    N_r = args.N_r
    snr = args.snr
    ridge_coeff = args.ridge_coeff
    rand_params = args.rand_params
    sig_sel_id = args.sig_sel_id
    rx_sel_id = args.rx_sel_id
    plot_level = args.plot_level
    figs_dir = args.figs_dir
    sig_noise = args.sig_noise

    sharp_bw = 10e6
    base_order_pos = 64
    base_order_neg = 0
    n_stage = 2
    fil_bank_mode = 2
    fil_mode = 2
    N_sig = 16
    N_r = 4
    snr = 10
    ridge_coeff = 1
    rand_params = True
    plot_level = 4
    sig_noise = False

    t = np.arange(0, n_samples) / fs  # Time vector
    freq = ((np.arange(1, n_samples + 1) / n_samples) - 0.5) * fs
    nfft = 2 ** np.ceil(np.log2(n_samples)).astype(int)

    if rand_params:
        sig_bw = 10e6 + 20e6 * rand(N_sig)
        sig_psd = (1/fs) * (1 * np.ones(N_sig) + 4 * rand(N_sig))
        sig_cf = (fs / 2) * (rand(N_sig) - 0.5)
        spatial_sig_rand_coef = 0.9
        spatial_sig = (1 - spatial_sig_rand_coef) * np.ones((N_r, N_sig)) + spatial_sig_rand_coef * rand(N_r, N_sig)
    else:
        N_sig = 8
        N_r = 4
        sig_bw = 1.0e+07 * np.array([2.5152, 1.5262, 1.1372, 2.8934, 2.9045, 2.5694, 2.7378, 1.2596])
        sig_psd = (1/fs) * np.array([3.0277, 3.1819, 1.0687, 3.6131, 3.8772, 2.4723, 4.7923, 4.8467])
        sig_cf = 1.0e+07 * np.array([-4.2928, -3.8345, 3.3524, -1.0737, 4.9128, -1.9313, -1.3051, 2.7511])
        spatial_sig = np.array([
            [0.5376, 0.6248, 0.1381, 0.8030, 0.3912, 0.9736, 0.1697, 0.9669],
            [0.7375, 0.1636, 0.4690, 0.9367, 0.2996, 0.2212, 0.4858, 0.9451],
            [0.9602, 0.8559, 0.1751, 0.6513, 0.9208, 0.6067, 0.2996, 0.5449],
            [0.6571, 0.5991, 0.1275, 0.3837, 0.6243, 0.9030, 0.2986, 0.7746]
        ])

        # N_sig = 2
        # N_r = 1
        # sig_bw = np.array([60e6, 2e6])
        # sig_psd = (1/fs) * np.array([1, 4])
        # sig_cf = np.array([0, 0])
        # spatial_sig = np.array([[1., 1.]])

        # N_sig = 5
        # N_r = 1
        # sig_bw = np.array([60e6, 2e6, 5e6, 6e6, 8e6])
        # sig_psd = (1/fs) * np.array([1, 4, 2, 1, 2])
        # sig_cf = np.array([0, 0, 10e6, 20e6, 50e6])
        # spatial_sig = np.array([[1, 1, 1, 1, 1]])

    sig_psd = sig_psd.astype(complex)
    spatial_sig = spatial_sig.astype(complex)

    #=================================================
    noise = randn(n_samples).astype(complex)
    rx = np.zeros((N_r, n_samples), dtype=complex)
    sigs = np.zeros((N_sig, n_samples), dtype=complex)

    for i in range(N_sig):
        fil_sig = firwin(1001, sig_bw[i] / fs)
        # sigs[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * sig_psd[i] * np.convolve(noise, fil_sig, mode='same')
        sigs[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * np.sqrt(sig_psd[i]*fs) * lfilter(fil_sig, 1, noise)
        rx += np.outer(spatial_sig[:, i], sigs[i, :])

        if sig_noise:
            yvar = np.mean(np.abs(sigs[i, :]) ** 2)
            wvar = yvar * (10 ** (-snr / 10))
            sigs[i, :] += np.sqrt(wvar / 2) * noise

    yvar = np.mean(np.abs(rx) ** 2, axis=1)
    wvar = yvar * (10 ** (-snr / 10))
    # rx += np.sqrt(wvar[:, None] / 2) * noise
    rx += np.outer(np.sqrt(wvar / 2), noise)

    if plot_level>=2:
        plt.figure()
        # plt.figure(figsize=(10,6))
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=1.0)
        plt.subplot(3, 1, 1)
        for i in range(N_sig):
            spectrum = fftshift(fft(sigs[i, :]))
            spectrum = 20 * np.log10(np.abs(spectrum))
            plt.plot(freq, spectrum, color=np.random.rand(3), linewidth=0.5)
        plt.title('Frequency spectrum of initial wideband signals')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')

        plt.subplot(3, 1, 2)
        spectrum = fftshift(fft(rx[rx_sel_id, :]))
        spectrum = 20 * np.log10(np.abs(spectrum))
        plt.plot(freq, spectrum, 'b-', linewidth=0.5)
        plt.title('Frequency spectrum of RX signal in a selected antenna')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')

        plt.subplot(3, 1, 3)
        spectrum = fftshift(fft(sigs[sig_sel_id, :]))
        spectrum = 20 * np.log10(np.abs(spectrum))
        plt.plot(freq, spectrum, 'r-', linewidth=0.5)
        plt.title('Frequency spectrum of the desired wideband signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')

        plt.savefig(figs_dir+'tx_rx_sigs.pdf', format='pdf')
        # plt.show(block=False)

    # frequencies, psd = welch(noise.astype(complex), fs, nperseg=1024)
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(frequencies, psd)
    # plt.title('Power Spectral Density (PSD) of Gaussian White Noise')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('PSD (V^2/Hz)')
    # plt.grid(True)
    # plt.show()
    # =================================================

    params = {'fs': fs, 'sharp_bw': sharp_bw, 'base_order_pos': base_order_pos,
              'base_order_neg': base_order_neg, 'n_stage': n_stage, 'us_rate': us_rate,
              'ds_rate': ds_rate, 'fil_bank_mode': fil_bank_mode, 'fil_mode': fil_mode,
              'snr': snr, 'ridge_coeff': ridge_coeff, 'sig_sel_id': sig_sel_id, 'rx_sel_id': rx_sel_id,
              'plot_level': plot_level, 'figs_dir': figs_dir}
    fir_separate_ins = fir_separate(params)

    fir_separate_ins.wiener_filter(rx, sigs)
    fir_separate_ins.wiener_filter_param(sig_bw, sig_psd, sig_cf, spatial_sig)
    fir_separate_ins.basis_filter(rx, sigs, sig_bw, sig_psd, sig_cf, spatial_sig)
    fir_separate_ins.basis_filter_param(sig_bw, sig_psd, sig_cf, spatial_sig)
    fir_separate_ins.visualize_errors()
    fir_separate_ins.visulalize_filter_delay()

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from filter_utils import *
from sklearn.linear_model import Ridge


class fir_separate(object):
    def __init__(self, rx, sigs, sig_bw, sig_amp, sig_cf, spatial_sig, params):

        self.rx = rx
        self.sigs = sigs

        params_default = {'fs': 200e6, 'sharp_bw': 10e6, 'base_order_pos': 64,
                          'base_order_neg': 0, 'n_stage': 1, 'us_rate': 2,
                          'ds_rate': 2, 'fil_bank_mode': 2, 'fil_mode': 1,
                          'snr': 10, 'ridge_coeff': 1, 'sig_sel_id': 0, 'rx_sel_id': 0, 'plot_level': 0}

        for param in params_default:
            if param in params:
                temp = params[param]
            else:
                temp = params_default[param]

            if param == 'fs':
                self.fs = temp
            elif param == 'sharp_bw':
                self.sharp_bw = temp
            elif param == 'base_order_pos':
                self.base_order_pos = temp
            elif param == 'base_order_neg':
                self.base_order_neg = temp
            elif param == 'n_stage':
                self.n_stage = temp
            elif param == 'us_rate':
                self.us_rate = temp
            elif param == 'ds_rate':
                self.ds_rate = temp
            elif param == 'fil_bank_mode':
                self.fil_bank_mode = temp
            elif param == 'fil_mode':
                self.fil_mode = temp
            elif param == 'snr':
                self.snr = temp
            elif param == 'ridge_coeff':
                self.ridge_coeff = temp
            elif param == 'sig_sel_id':
                self.sig_sel_id = temp
            elif param == 'rx_sel_id':
                self.rx_sel_id = temp
            elif param == 'plot_level':
                self.plot_level = temp

        self.N_sig = sigs.shape[0]
        self.N_r = rx.shape[0]
        self.n_samples = sigs.shape[1]

        self.sig_bw = sig_bw
        self.sig_amp = sig_amp
        self.sig_cf = sig_cf
        self.spatial_sig = spatial_sig

        self.sharp_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.sharp_order_neg = self.base_order_neg * (2 ** self.n_stage)
        self.wiener_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.wiener_order_neg = self.base_order_neg * (2 ** self.n_stage)

        self.t = np.arange(0, self.n_samples) / self.fs

        self.nfft = 2 ** np.ceil(np.log2(self.n_samples)).astype(int)

        self.grp_dly_base = (self.base_order_pos // 2)
        self.grp_dly_sharp = (self.sharp_order_pos // 2)

        self.wiener_errs = np.zeros(self.N_sig)
        self.basis_errs = np.zeros(self.N_sig)

        self.om = np.linspace(-np.pi, np.pi, self.n_samples)
        self.freq = ((np.arange(1, self.n_samples + 1) / self.n_samples) - 0.5) * self.fs

        if self.fil_bank_mode == 1:
            self.fil_bank_num = int(self.fs / self.sharp_bw)
            self.fil_cf = (-self.fs / 2) + (self.sharp_bw / 2) + np.linspace(0, self.fil_bank_num - 1, self.fil_bank_num) * self.sharp_bw
        elif self.fil_bank_mode == 2:
            self.fil_bank_num = self.N_sig
            self.fil_cf = self.sig_cf.copy()

        self.fil_base = [None] * self.fil_bank_num
        self.fil_sharp = [None] * self.fil_bank_num

        for i in range(self.fil_bank_num):
            if self.fil_bank_mode == 1:
                fil_bw_base = self.sharp_bw
            elif self.fil_bank_mode == 2:
                fil_bw_base = self.sig_bw[i]
            self.fil_base[i] = firwin(self.base_order_pos + 1, fil_bw_base * (2 ** self.n_stage) / self.fs)
            self.fil_sharp[i] = firwin(self.sharp_order_pos + 1, fil_bw_base / self.fs)

        self.fil_bank = [None] * self.fil_bank_num
        for i in range(self.fil_bank_num):
            t_fil = self.t[:len(self.fil_sharp[i])]
            self.fil_bank[i] = np.exp(2 * np.pi * 1j * self.fil_cf[i] * t_fil) * self.fil_sharp[i]

        if self.plot_level>=2:
            plt.figure()
            for i in range(self.fil_bank_num):
                if i % 1 == 0:
                    w, h = freqz(self.fil_bank[i], worN=self.om)
                    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), label=f'Filter {i + 1}')
            plt.title('Frequency response of selected filters in the filter bank')
            plt.xlabel('Normalized Frequency (xpi rad/sample)')
            plt.ylabel('Magnitude (dB)')
            # plt.legend()
            plt.show()

        self.sig_bank = [[None] * self.N_r for _ in range(self.fil_bank_num)]
        for i in range(self.fil_bank_num):
            for j in range(self.N_r):
                plot_procedure = i == int(3 * self.fil_bank_num / 4) and j == self.rx_sel_id and self.plot_level >= 5
                if self.fil_mode == 1:
                    # sig_bank[i][j] = np.convolve(rx[j, :], fil_bank[i], mode='same')
                    self.sig_bank[i][j] = lfilter(self.fil_bank[i], 1, self.rx[j, :])
                    self.filter_delay = self.grp_dly_sharp
                elif self.fil_mode == 2:
                    self.sig_bank[i][j], self.filter_delay = basis_fir_us(rx[j, :], self.fil_base[i], self.t, self.freq, self.fil_cf[i], self.n_stage,
                                                                          self.us_rate, plot_procedure)
                elif self.fil_mode == 3:
                    self.sig_bank[i][j], self.filter_delay = basis_fir_ds_us(rx[j, :], self.fil_base[i], self.t, self.freq, self.fil_cf[i], self.n_stage,
                                                                             self.ds_rate, self.us_rate, plot_procedure)
                else:
                    raise ValueError('Invalid Filtering mode %d' % self.fil_mode)

                self.sig_bank[i][j] = self.sig_bank[i][j].astype(complex)

        print(f'Total group delay for filtering: {self.filter_delay}')

        if self.plot_level >= 3:
            plt.figure()
            for i in range(self.fil_bank_num):
                if i % 1 == 0:
                    spectrum = fftshift(fft(self.sig_bank[i][self.rx_sel_id]))
                    spectrum = 20 * np.log10(np.abs(spectrum))
                    plt.plot(self.freq, spectrum, color=np.random.rand(3), linewidth=0.5)
            plt.title('Frequency spectrum of the signal bank filtered using filter bank')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.show()


    def wiener_filter(self):
        rx_dly = self.rx
        self.fil_wiener_single = [[None] * self.N_r for _ in range(self.N_sig)]

        if self.N_r <= 1:
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = wiener_fir(self.rx, self.sigs[i, :].reshape((1, -1)), self.wiener_order_pos,
                                                         self.wiener_order_neg).reshape(-1)
        else:
            fil_wiener = wiener_fir_vector(self.rx, self.sigs, self.wiener_order_pos, self.wiener_order_neg)
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::self.N_r]

        for i in range(self.N_sig):
            sig_filtered_wiener = np.zeros_like(self.t, dtype=complex)
            for j in range(self.N_r):
                # sig_filtered_wiener += np.convolve(rx_dly[j, :], fil_wiener_single[i][j], mode='same')
                sig_filtered_wiener += lfilter(self.fil_wiener_single[i][j], 1, rx_dly[j, :])

            time_delay = extract_delay(sig_filtered_wiener, self.sigs[i, :], self.plot_level >= 5)
            print(f'Time delay between the signal and its Wiener filtered version for {i + 1}: {time_delay} samples')

            sig_filtered_wiener_adj, signal_adj, mse, err2sig_ratio = time_adjust(sig_filtered_wiener, self.sigs[i, :],
                                                                                  time_delay)
            print(
                f'Error to signal ratio for the estimation of the main signal using Wiener filter for {i + 1}: {err2sig_ratio}')
            self.wiener_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(self.n_samples // 2, self.n_samples // 2 + 500)
                plt.plot(self.t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                plt.plot(self.t[index], np.abs(sig_filtered_wiener_adj[index]), 'b-', linewidth=0.5)
                plt.title('Signal and its recovered wiener filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                plt.show()

        if self.plot_level >= 3:
            plt.figure()
            w, h = freqz(self.fil_wiener_single[self.sig_sel_id][self.rx_sel_id], worN=self.om)
            plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
            plt.title('Frequency response of the Wiener filter for the selected TX signal and RX antenna')
            plt.xlabel('Normalized Frequency (xpi rad/sample)')
            plt.ylabel('Magnitude (dB)')
            plt.show()


    def basis_filter(self):
        shift = self.filter_delay
        sig_bank_mat = np.zeros((self.n_samples - shift, self.fil_bank_num * self.N_r), dtype=complex)
        for j in range(self.N_r):
            for i in range(self.fil_bank_num):
                sig_bank_mat[:, (j * self.fil_bank_num + i)] = self.sig_bank[i][j][shift:]
        b = np.copy(self.sigs[:, :self.n_samples - shift].T)

        for i in range(self.N_sig):
            # # self.sig_bank_coeffs = np.linalg.lstsq(sig_bank_mat.T @ sig_bank_mat + self.ridge_coeff * np.eye(self.fil_bank_num * self.N_r), sig_bank_mat.T @ b[:,i],
            # #                 rcond=None)[0]
            # self.sig_bank_coeffs = np.linalg.inv(sig_bank_mat.T @ sig_bank_mat + (self.ridge_coeff * np.eye(self.fil_bank_num * self.N_r))) @ (sig_bank_mat.T) @ b[:,i]
            # sig_filtered_base = (sig_bank_mat @ self.sig_bank_coeffs).T

            sig_bank_mat_real = np.real(sig_bank_mat)
            sig_bank_mat_imag = np.imag(sig_bank_mat)
            sig_bank_mat_combined = np.hstack([sig_bank_mat_real, sig_bank_mat_imag])
            b_real = np.real(b[:, i])
            b_imag = np.imag(b[:, i])
            ridge_real = Ridge(alpha=self.ridge_coeff)
            ridge_imag = Ridge(alpha=self.ridge_coeff)
            ridge_real.fit(sig_bank_mat_combined, b_real)
            ridge_imag.fit(sig_bank_mat_combined, b_imag)

            sig_filtered_base_real = ridge_real.predict(sig_bank_mat_combined)
            sig_filtered_base_imag = ridge_imag.predict(sig_bank_mat_combined)
            self.sig_filtered_base = sig_filtered_base_real + 1j * sig_filtered_base_imag
            self.sig_filtered_base = self.sig_filtered_base.T

            sig_bank_coeffs_real = ridge_real.coef_
            sig_bank_coeffs_imag = ridge_imag.coef_
            sig_bank_coeffs_real_real = sig_bank_coeffs_real[:sig_bank_mat.shape[1]]
            sig_bank_coeffs_real_imag = sig_bank_coeffs_real[sig_bank_mat.shape[1]:]
            sig_bank_coeffs_imag_real = sig_bank_coeffs_imag[:sig_bank_mat.shape[1]]
            sig_bank_coeffs_imag_imag = sig_bank_coeffs_imag[sig_bank_mat.shape[1]:]

            sig_bank_multiplied = np.multiply(sig_bank_coeffs_real_real.reshape((1, -1)), sig_bank_mat_real) \
                                  + np.multiply(sig_bank_coeffs_real_imag.reshape((1, -1)), sig_bank_mat_imag) \
                                  + np.multiply(sig_bank_coeffs_imag_real.reshape((1, -1)), sig_bank_mat_real * 1j) \
                                  + np.multiply(sig_bank_coeffs_imag_imag.reshape((1, -1)), sig_bank_mat_imag * 1j)
            sig_bank_coeffs_mat = np.divide(sig_bank_multiplied, sig_bank_mat)
            self.sig_bank_coeffs = np.mean(sig_bank_coeffs_mat, axis=0).reshape(-1)
            # var_mat = (sig_bank_coeffs_mat-np.tile(self.sig_bank_coeffs, (sig_bank_coeffs_mat.shape[0],1)))**2
            # print(np.mean(var_mat, axis=0))

            time_delay = extract_delay(self.sig_filtered_base, self.sigs[i, :self.n_samples - shift], self.plot_level >= 5)
            print(f'Time delay between the signal and its basis filtered version for {i + 1}: {time_delay} samples')
            # time_delay = 0
            sig_filtered_base_adj, signal_adj, mse, err2sig_ratio = time_adjust(self.sig_filtered_base,
                                                                                self.sigs[i, :self.n_samples - shift], time_delay)
            print(
                f'Error to signal ratio for the estimation of the main signal using basis filter for {i + 1}: {err2sig_ratio}')
            self.basis_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(self.n_samples // 2, self.n_samples // 2 + 500)
                plt.plot(self.t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                # plt.plot(t[index], np.abs(self.sigs[i, index]), 'b-', linewidth=0.5)
                plt.plot(self.t[index], np.abs(sig_filtered_base_adj[index]), 'b-', linewidth=0.5)
                # plt.plot(t[index], np.abs(self.sig_filtered_base[i, index]),'r-', linewidth=0.5)
                plt.title('Signal and its recovered basis filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                plt.show()

            if i == self.sig_sel_id and self.plot_level >= 3:
                freq_range = self.fil_cf
                coeffs_range = np.arange(self.rx_sel_id * self.fil_bank_num, self.rx_sel_id * self.fil_bank_num + self.fil_bank_num)
                coeffs = np.abs(self.sig_bank_coeffs[coeffs_range])

                if self.fil_bank_mode == 2:
                    sorted_indices = np.argsort(freq_range)
                    freq_range = freq_range[sorted_indices]
                    coeffs = coeffs[sorted_indices]

                # Plot the basis filters coefficients
                plt.figure()
                plt.plot(freq_range, coeffs, 'b-')
                plt.title('Basis filters coefficients for the selected signal for each center frequency')
                plt.xlabel('Basis Filter Center Frequency (Hz)')
                plt.ylabel('Coefficient')
                plt.show()


    def visualize_errors(self):

        if self.plot_level >= 1:
            # plt.figure()
            # plt.scatter(np.arange(1, self.N_sig + 1), self.wiener_errs, color='b', label='Wiener')
            # plt.scatter(np.arange(1, self.N_sig + 1), self.basis_errs, color='r', label='Basis')
            # plt.legend()
            # plt.title('Basis and Wiener errors')
            # plt.xlabel('Signal Index')
            # plt.ylabel('Error')
            # plt.show()

            plt.figure()
            plt.scatter(np.arange(1, self.N_sig + 1), self.basis_errs / self.wiener_errs, color='b', label='B/W')
            plt.scatter(np.arange(1, self.N_sig + 1), self.wiener_errs / self.basis_errs, color='r', label='W/B')
            plt.legend()
            plt.title('Wiener over basis and basis over wiener errors ratio')
            plt.xlabel('Signal Index')
            plt.ylabel('Ratio')
            plt.show()

        print(f'Mean error to signal ratio for Wiener filtering: {np.mean(self.wiener_errs)}')
        print(f'Mean error to signal ratio for Basis filtering: {np.mean(self.basis_errs)}')

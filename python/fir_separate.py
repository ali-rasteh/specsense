import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from filter_utils import *
from sklearn.linear_model import Ridge


class fir_separate(object):
    def __init__(self, params):

        params_default = {'fs': 200e6, 'sharp_bw': 10e6, 'base_order_pos': 64,
                          'base_order_neg': 0, 'n_stage': 1, 'us_rate': 2,
                          'ds_rate': 2, 'fil_bank_mode': 2, 'fil_mode': 1,
                          'snr': 10, 'ridge_coeff': 1, 'sig_sel_id': 0, 'rx_sel_id': 0,
                          'plot_level': 0, 'figs_dir':'./figs/'}

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
            elif param == 'figs_dir':
                self.figs_dir = temp

        self.sharp_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.sharp_order_neg = self.base_order_neg * (2 ** self.n_stage)
        self.wiener_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.wiener_order_neg = self.base_order_neg * (2 ** self.n_stage)

        self.grp_dly_base = (self.base_order_pos // 2)
        self.grp_dly_sharp = (self.sharp_order_pos // 2)


    def wiener_filter(self, rx, sigs):
        print('\n\nBeginning to design the optimal wiener filter using the rx and desired signals.')

        N_sig = sigs.shape[0]
        N_r = rx.shape[0]
        n_samples = sigs.shape[1]

        t = np.arange(0, n_samples) / self.fs
        nfft = 2 ** np.ceil(np.log2(n_samples)).astype(int)

        om = np.linspace(-np.pi, np.pi, n_samples)
        freq = ((np.arange(1, n_samples + 1) / n_samples) - 0.5) * self.fs

        self.wiener_errs = np.zeros(N_sig)

        rx_dly = rx.copy()
        self.fil_wiener_single = [[None] * N_r for _ in range(N_sig)]

        if N_r <= 1:
            for i in range(N_sig):
                for j in range(N_r):
                    self.fil_wiener_single[i][j] = wiener_fir(rx, sigs[i, :].reshape((1, -1)), self.wiener_order_pos,
                                                              self.wiener_order_neg).reshape(-1)
        else:
            fil_wiener = wiener_fir_vector(rx, sigs, self.wiener_order_pos, self.wiener_order_neg)
            for i in range(N_sig):
                for j in range(N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::N_r]

        for i in range(N_sig):
            sig_filtered_wiener = np.zeros_like(t, dtype=complex)
            for j in range(N_r):
                # sig_filtered_wiener += np.convolve(rx_dly[j, :], fil_wiener_single[i][j], mode='same')
                sig_filtered_wiener += lfilter(self.fil_wiener_single[i][j], 1, rx_dly[j, :])

            time_delay = extract_delay(sig_filtered_wiener, sigs[i, :], self.plot_level >= 5)
            print(f'Time delay between the signal and its Wiener filtered version for signal {i + 1}: {time_delay} samples')

            sig_filtered_wiener_adj, signal_adj, mse, err2sig_ratio = time_adjust(sig_filtered_wiener, sigs[i, :],
                                                                                  time_delay)
            print(
                f'Error to signal ratio for the estimation of the main signal using Wiener filter for signal {i + 1}: {err2sig_ratio}')
            self.wiener_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(n_samples // 2, n_samples // 2 + 500)
                plt.plot(t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                plt.plot(t[index], np.abs(sig_filtered_wiener_adj[index]), 'b-', linewidth=0.5)
                plt.title('Signal and its recovered wiener filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                # plt.show(block=False)

        if self.plot_level >= 3:
            # plt.figure()
            # w, h = freqz(self.fil_wiener_single[self.sig_sel_id][self.rx_sel_id], worN=om)
            # plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
            # plt.title('Frequency response of the Wiener filter \n for the selected TX signal and RX antenna')
            # plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            # plt.ylabel('Magnitude (dB)')
            # # plt.show(block=False)

            plt.figure()
            plt.subplots_adjust(wspace=0.5, hspace=1.0)
            for rx_id in range(N_r):
                plt.subplot(N_r,1,rx_id+1)
                w, h = freqz(self.fil_wiener_single[self.sig_sel_id][rx_id], worN=om)
                plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
                plt.title('Selected TX signal, and RX antenna {}'.format(rx_id+1))
                if rx_id == 1:
                    plt.ylabel('Magnitude (dB)')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.savefig(self.figs_dir + 'wiener_filters.pdf', format='pdf')
            # plt.show(block=False)


    def wiener_filter_param(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        print('\n\nBeginning to design the optimal wiener filter using paramters.')

        N_sig = spatial_sig.shape[1]
        N_r = spatial_sig.shape[0]
        n_samples = 2**13

        t = np.arange(0, n_samples) / self.fs
        nfft = 2 ** np.ceil(np.log2(n_samples)).astype(int)

        om = np.linspace(-np.pi, np.pi, n_samples)
        freq = ((np.arange(1, n_samples + 1) / n_samples) - 0.5) * self.fs

        self.wiener_errs_param = np.zeros(N_sig)

        self.fil_wiener_single = [[None] * N_r for _ in range(N_sig)]

        if N_r <= 1:
            for i in range(N_sig):
                for j in range(N_r):
                    self.fil_wiener_single[i][j] = wiener_fir_param(sig_bw, sig_psd, sig_cf, spatial_sig, self.snr, self.wiener_order_pos,
                                                              self.wiener_order_neg).reshape(-1)
        else:
            fil_wiener = wiener_fir_vector_param(sig_bw, sig_psd, sig_cf, spatial_sig, self.snr, self.wiener_order_pos, self.wiener_order_neg)
            for i in range(N_sig):
                for j in range(N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::N_r]

        if self.plot_level >= 3:
            plt.figure()
            w, h = freqz(self.fil_wiener_single[self.sig_sel_id][self.rx_sel_id], worN=om)
            plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
            plt.title('Frequency response of the parametric Wiener filter \n for the selected TX signal and RX antenna')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.ylabel('Magnitude (dB)')
            # plt.show(block=False)


    def basis_filter(self, rx, sigs, sig_bw, sig_psd, sig_cf, spatial_sig):

        print('\n\nBeginning to design the optimal basis filters using the rx and desired signals data.')

        N_sig = sigs.shape[0]
        N_r = rx.shape[0]
        n_samples = sigs.shape[1]

        t = np.arange(0, n_samples) / self.fs
        nfft = 2 ** np.ceil(np.log2(n_samples)).astype(int)

        om = np.linspace(-np.pi, np.pi, n_samples)
        freq = ((np.arange(1, n_samples + 1) / n_samples) - 0.5) * self.fs

        self.basis_errs = np.zeros(N_sig)

        if self.fil_bank_mode == 1:
            self.fil_bank_num = int(self.fs / self.sharp_bw)
            self.fil_cf = (-self.fs / 2) + (self.sharp_bw / 2) + np.linspace(0, self.fil_bank_num - 1, self.fil_bank_num) * self.sharp_bw
        elif self.fil_bank_mode == 2:
            self.fil_bank_num = N_sig
            self.fil_cf = sig_cf.copy()

        self.fil_base = [None] * self.fil_bank_num
        self.fil_sharp = [None] * self.fil_bank_num

        for i in range(self.fil_bank_num):
            if self.fil_bank_mode == 1:
                fil_bw_base = self.sharp_bw
            elif self.fil_bank_mode == 2:
                fil_bw_base = sig_bw[i]
            self.fil_base[i] = firwin(self.base_order_pos + 1, fil_bw_base * (2 ** self.n_stage) / self.fs)
            self.fil_sharp[i] = firwin(self.sharp_order_pos + 1, fil_bw_base / self.fs)

        self.fil_bank = [None] * self.fil_bank_num
        for i in range(self.fil_bank_num):
            t_fil = t[:len(self.fil_sharp[i])]
            self.fil_bank[i] = np.exp(2 * np.pi * 1j * self.fil_cf[i] * t_fil) * self.fil_sharp[i]

        if self.plot_level>=2:
            plt.figure()
            for i in range(self.fil_bank_num):
                if i % 1 == 0:
                    w, h = freqz(self.fil_bank[i], worN=om)
                    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), label=f'Filter {i + 1}')
            plt.title('Frequency response of basis filters in the filter bank')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(self.figs_dir + 'basis_filters.pdf', format='pdf')
            # plt.legend()
            # plt.show(block=False)

        self.sig_bank = [[None] * N_r for _ in range(self.fil_bank_num)]
        for i in range(self.fil_bank_num):
            for j in range(N_r):
                plot_procedure = i == int(3 * self.fil_bank_num / 4) and j == self.rx_sel_id and self.plot_level >= 5
                if self.fil_mode == 1:
                    # sig_bank[i][j] = np.convolve(rx[j, :], fil_bank[i], mode='same')
                    self.sig_bank[i][j] = lfilter(self.fil_bank[i], 1, rx[j, :])
                    self.filter_delay = self.grp_dly_sharp
                elif self.fil_mode == 2:
                    self.sig_bank[i][j], self.filter_delay = basis_fir_us(rx[j, :], self.fil_base[i], t, freq, self.fil_cf[i], self.n_stage,
                                                                          self.us_rate, plot_procedure)
                elif self.fil_mode == 3:
                    self.sig_bank[i][j], self.filter_delay = basis_fir_ds_us(rx[j, :], self.fil_base[i], t, freq, self.fil_cf[i], self.n_stage,
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
                    plt.plot(freq, spectrum, color=np.random.rand(3), linewidth=0.5)
            plt.title('Frequency spectrum of the signal bank filtered using the filter bank')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(self.figs_dir + 'signal_bank.pdf', format='pdf')
            # plt.show(block=False)


        shift = self.filter_delay
        sig_bank_mat = np.zeros((n_samples - shift, self.fil_bank_num * N_r), dtype=complex)
        for j in range(N_r):
            for i in range(self.fil_bank_num):
                sig_bank_mat[:, (j * self.fil_bank_num + i)] = self.sig_bank[i][j][shift:]
        b = np.copy(sigs[:, :n_samples - shift].T)

        for i in range(N_sig):
            # # self.sig_bank_coeffs = np.linalg.lstsq(sig_bank_mat.T @ sig_bank_mat + self.ridge_coeff * np.eye(self.fil_bank_num * N_r), sig_bank_mat.T @ b[:,i],
            # #                 rcond=None)[0]
            # self.sig_bank_coeffs = np.linalg.inv(sig_bank_mat.T @ sig_bank_mat + (self.ridge_coeff * np.eye(self.fil_bank_num * N_r))) @ (sig_bank_mat.T) @ b[:,i]
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

            time_delay = extract_delay(self.sig_filtered_base, sigs[i, :n_samples - shift], self.plot_level >= 5)
            print(f'Time delay between the signal and its basis filtered version for signal {i + 1}: {time_delay} samples')
            # time_delay = 0
            sig_filtered_base_adj, signal_adj, mse, err2sig_ratio = time_adjust(self.sig_filtered_base,
                                                                                sigs[i, :n_samples - shift], time_delay)
            print(
                f'Error to signal ratio for the estimation of the main signal using basis filter for signal {i + 1}: {err2sig_ratio}')
            self.basis_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(n_samples // 2, n_samples // 2 + 500)
                plt.plot(t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                # plt.plot(t[index], np.abs(self.sigs[i, index]), 'b-', linewidth=0.5)
                plt.plot(t[index], np.abs(sig_filtered_base_adj[index]), 'b-', linewidth=0.5)
                # plt.plot(t[index], np.abs(self.sig_filtered_base[i, index]),'r-', linewidth=0.5)
                plt.title('Signal and its recovered basis filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                # plt.show(block=False)

            if i == self.sig_sel_id and self.plot_level >= 3:
                freq_range = self.fil_cf
                coeffs_range = np.arange(self.rx_sel_id * self.fil_bank_num, self.rx_sel_id * self.fil_bank_num + self.fil_bank_num)
                coeffs_mag = np.abs(self.sig_bank_coeffs[coeffs_range])
                coeffs_ang = np.angle(self.sig_bank_coeffs[coeffs_range])

                if self.fil_bank_mode == 2:
                    sorted_indices = np.argsort(freq_range)
                    freq_range = freq_range[sorted_indices]
                    coeffs_mag = coeffs_mag[sorted_indices]
                    coeffs_ang = coeffs_ang[sorted_indices]

                # Plot the basis filters coefficients
                plt.figure()
                plt.subplots_adjust(wspace=0.5, hspace=0.5)
                plt.subplot(2,1,1)
                plt.title('Basis Filters Coefficients For The Selected Signal')
                plt.plot(freq_range, coeffs_mag, 'b-')
                plt.ylabel('Coefficient Magnitude')
                plt.subplot(2, 1, 2)
                plt.plot(freq_range, coeffs_ang, 'b-')
                plt.xlabel('Basis Filter Center Frequency (Hz)')
                plt.ylabel('Coefficient Angle (Rad)')
                plt.savefig(self.figs_dir + 'basis_coeffs.pdf', format='pdf')
                # plt.show(block=False)


    def basis_filter_param(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        print('\n\nBeginning to design the optimal basis filter using paramters.')



    def visualize_errors(self):

        if self.plot_level >= 1:
            print('\n\nReporting errors of Wiener and basis filtering.')

            # plt.figure()
            # plt.scatter(np.arange(1, self.N_sig + 1), self.wiener_errs, color='b', label='Wiener')
            # plt.scatter(np.arange(1, self.N_sig + 1), self.basis_errs, color='r', label='Basis')
            # plt.legend()
            # plt.title('Basis and Wiener errors')
            # plt.xlabel('Signal Index')
            # plt.ylabel('Error')
            # # plt.show(block=False)

            if self.basis_errs.shape[0] != self.wiener_errs.shape[0]:
                raise ValueError('Filtering errors size mismatch between wiener and basis filtering {}, {}'.format(self.wiener_errs.shape[0], self.basis_errs.shape[0]))
            else:
                N_sig = self.basis_errs.shape[0]

            plt.figure()
            plt.scatter(np.arange(1, N_sig + 1), self.basis_errs / self.wiener_errs, color='b', label='B/W')
            plt.scatter(np.arange(1, N_sig + 1), self.wiener_errs / self.basis_errs, color='r', label='W/B')
            plt.legend()
            plt.title('Wiener over basis and basis over wiener errors ratio')
            plt.xlabel('Signal Index')
            plt.ylabel('Ratio')
            # plt.show(block=False)

        print(f'Mean error to signal ratio for Wiener filtering: {np.mean(self.wiener_errs)}')
        print(f'Mean error to signal ratio for Basis filtering: {np.mean(self.basis_errs)}')


    def visulalize_filter_delay(self):
        if self.plot_level >= 1:
            plt.figure()
            S = np.array(range(0, 8))
            delay_1 = (3 * 2 ** S - 2) / (2 ** S)
            delay_2 = (2 ** (S + 1) - 1) / (2 ** S)
            plt.plot(S, delay_1, color='b', label='Signal up/down-sampling')
            plt.plot(S, delay_2, color='r', label='Filter up-sampling')
            plt.title('Comaprison of proposed methods and conventional filtering delays')
            plt.xlabel('S (number of stages)')
            plt.ylabel('Normalized delay relative to conventional filtering')
            plt.legend()
            plt.savefig(self.figs_dir + 'filter_delay.pdf', format='pdf')
            plt.show()



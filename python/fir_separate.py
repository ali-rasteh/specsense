from backend import *
from backend import be_np as np, be_scp as scipy
from filter_utils import Filter_Utils




class Fir_Separate(Filter_Utils):
    def __init__(self, params):
        super().__init__(params)

        self.sharp_bw = params.sharp_bw
        self.base_order_pos = params.base_order_pos
        self.base_order_neg = params.base_order_neg
        self.n_stage = params.n_stage
        self.us_rate = params.us_rate
        self.ds_rate = params.ds_rate
        self.fil_bank_mode = params.fil_bank_mode
        self.fil_mode = params.fil_mode
        self.ridge_coeff = params.ridge_coeff
        self.fo_f_id = params.fo_f_id
        self.snr_f_id = params.snr_f_id
        self.N_sig_f_id = params.N_sig_f_id
        self.N_r_f_id = params.N_r_f_id
        self.plot_mean = params.plot_mean

        self.sharp_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.sharp_order_neg = self.base_order_neg * (2 ** self.n_stage)
        self.wiener_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.wiener_order_neg = self.base_order_neg * (2 ** self.n_stage)

        self.grp_dly_base = (self.base_order_pos // 2)
        self.grp_dly_sharp = (self.sharp_order_pos // 2)

        self.basis_fil_ridge_real = None
        self.basis_fil_ridge_imag = None
        self.aoa = None

        self.print('Initialized the fir_separate class instance.',2)


    def basis_filter_design(self, rx, sigs, sig_bw, sig_cf):
        self.print('Beginning to design the optimal basis filters using the rx and desired signals data.',2)

        if self.fil_bank_mode == 1:
            self.fil_bank_num = int(self.fs / self.sharp_bw)
            self.fil_cf = (-self.fs / 2) + (self.sharp_bw / 2) + np.linspace(0, self.fil_bank_num - 1, self.fil_bank_num) * self.sharp_bw
        elif self.fil_bank_mode == 2:
            self.fil_bank_num = self.N_sig
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
            t_fil = self.t[:len(self.fil_sharp[i])]
            self.fil_bank[i] = np.exp(2 * np.pi * 1j * self.fil_cf[i] * t_fil) * self.fil_sharp[i]

        if self.plot_level>=2:
            plt.figure()
            for i in range(self.fil_bank_num):
                if i % 1 == 0:
                    w, h = freqz(self.fil_bank[i], worN=self.om)
                    plt.plot(w / np.pi, self.lin_to_db(np.abs(h), mode='mag'), label=f'Filter {i + 1}')
            plt.title('Frequency response of basis filters in the filter bank')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(os.path.join(self.figs_dir, 'basis_filters.pdf'), format='pdf')
            # plt.legend()
            # plt.show(block=False)

        self.basis_fil_ridge_real = [Ridge(alpha=self.ridge_coeff) for _ in range(self.N_sig)]
        self.basis_fil_ridge_imag = [Ridge(alpha=self.ridge_coeff) for _ in range(self.N_sig)]

        self.basis_filter_apply(rx, sigs, mode='train')


    def basis_filter_apply(self, rx, sigs, mode='train'):
        self.print('Beginning to apply the designed basis filter on the rx signal in mode: {}.'.format(mode),2)

        self.sig_bank = [[None] * self.N_r for _ in range(self.fil_bank_num)]
        self.basis_errs = np.zeros(self.N_sig)

        for i in range(self.fil_bank_num):
            for j in range(self.N_r):
                plot_procedure = i == int(3 * self.fil_bank_num / 4) and j == self.rx_sel_id and self.plot_level >= 5
                if self.fil_mode == 1:
                    # sig_bank[i][j] = np.convolve(rx[j, :], fil_bank[i], mode='same')
                    self.sig_bank[i][j] = lfilter(self.fil_bank[i], np.array([1]), rx[j, :])
                    self.filter_delay = self.grp_dly_sharp
                elif self.fil_mode == 2:
                    self.sig_bank[i][j], self.filter_delay = self.basis_fir_us(rx[j, :], self.fil_base[i], self.t, self.freq,
                                                                                     self.fil_cf[i], self.n_stage,
                                                                                     self.us_rate, plot_procedure)
                elif self.fil_mode == 3:
                    self.sig_bank[i][j], self.filter_delay = self.basis_fir_ds_us(rx[j, :], self.fil_base[i], self.t,
                                                                                        self.freq, self.fil_cf[i], self.n_stage,
                                                                                        self.ds_rate, self.us_rate, plot_procedure)
                else:
                    raise ValueError('Invalid Filtering mode %d' % self.fil_mode)

                self.sig_bank[i][j] = self.sig_bank[i][j].astype(complex)

        self.print(f'Total group delay for filtering: {self.filter_delay}',4)

        if self.plot_level >= 3:
            plt.figure()
            for i in range(self.fil_bank_num):
                if i % 1 == 0:
                    spectrum = fftshift(fft(self.sig_bank[i][self.rx_sel_id]))
                    spectrum = self.lin_to_db(np.abs(spectrum), mode='mag')
                    plt.plot(self.freq, spectrum, color=rand(3), linewidth=0.5)
            plt.title('Frequency spectrum of the signal bank filtered using the filter bank')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(os.path.join(self.figs_dir, 'signal_bank.pdf'), format='pdf')
            # plt.show(block=False)

        shift = self.filter_delay
        sig_bank_mat = np.zeros((self.n_samples - shift, self.fil_bank_num * self.N_r), dtype=complex)
        for j in range(self.N_r):
            for i in range(self.fil_bank_num):
                sig_bank_mat[:, (j * self.fil_bank_num + i)] = self.sig_bank[i][j][shift:]
        b = np.copy(sigs[:, :self.n_samples - shift].T)

        sig_bank_mat = self.numpy_transfer(sig_bank_mat, dst='numpy')
        sig_bank_mat_real = numpy.real(sig_bank_mat)
        sig_bank_mat_imag = numpy.imag(sig_bank_mat)
        sig_bank_mat_combined = numpy.hstack([sig_bank_mat_real, sig_bank_mat_imag])
        b = self.numpy_transfer(b, dst='numpy')

        for i in range(self.N_sig):
            # # self.sig_bank_coeffs = np.linalg.lstsq(sig_bank_mat.T @ sig_bank_mat + self.ridge_coeff * np.eye(self.fil_bank_num * N_r), sig_bank_mat.T @ b[:,i],
            # #                 rcond=None)[0]
            # self.sig_bank_coeffs = np.linalg.inv(sig_bank_mat.T @ sig_bank_mat + (self.ridge_coeff * np.eye(self.fil_bank_num * N_r))) @ (sig_bank_mat.T) @ b[:,i]
            # sig_filtered_base = (sig_bank_mat @ self.sig_bank_coeffs).T

            if mode=='train':
                b_real = numpy.real(b[:, i])
                b_imag = numpy.imag(b[:, i])
                self.basis_fil_ridge_real[i].fit(sig_bank_mat_combined, b_real)
                self.basis_fil_ridge_imag[i].fit(sig_bank_mat_combined, b_imag)

                sig_bank_coeffs_real = self.basis_fil_ridge_real[i].coef_
                sig_bank_coeffs_imag = self.basis_fil_ridge_imag[i].coef_
                sig_bank_coeffs_real_real = sig_bank_coeffs_real[:sig_bank_mat.shape[1]]
                sig_bank_coeffs_real_imag = sig_bank_coeffs_real[sig_bank_mat.shape[1]:]
                sig_bank_coeffs_imag_real = sig_bank_coeffs_imag[:sig_bank_mat.shape[1]]
                sig_bank_coeffs_imag_imag = sig_bank_coeffs_imag[sig_bank_mat.shape[1]:]

                sig_bank_multiplied = numpy.multiply(sig_bank_coeffs_real_real.reshape((1, -1)), sig_bank_mat_real) \
                                      + numpy.multiply(sig_bank_coeffs_real_imag.reshape((1, -1)), sig_bank_mat_imag) \
                                      + numpy.multiply(sig_bank_coeffs_imag_real.reshape((1, -1)), sig_bank_mat_real * 1j) \
                                      + numpy.multiply(sig_bank_coeffs_imag_imag.reshape((1, -1)), sig_bank_mat_imag * 1j)
                sig_bank_coeffs_mat = numpy.divide(sig_bank_multiplied, sig_bank_mat)
                self.sig_bank_coeffs = numpy.mean(sig_bank_coeffs_mat, axis=0).reshape(-1)
                # var_mat = (sig_bank_coeffs_mat-numpy.tile(self.sig_bank_coeffs, (sig_bank_coeffs_mat.shape[0],1)))**2
                # print(numpy.mean(var_mat, axis=0))

                if i == self.sig_sel_id and self.plot_level >= 3:
                    freq_range = self.fil_cf
                    coeffs_range = np.arange(self.rx_sel_id * self.fil_bank_num,
                                             self.rx_sel_id * self.fil_bank_num + self.fil_bank_num)
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
                    plt.subplot(2, 1, 1)
                    plt.title('Basis Filters Coefficients For The Selected Signal')
                    plt.plot(freq_range, coeffs_mag, 'b-')
                    plt.ylabel('Coefficient Magnitude')
                    plt.subplot(2, 1, 2)
                    plt.plot(freq_range, coeffs_ang, 'b-')
                    plt.xlabel('Basis Filter Center Frequency (Hz)')
                    plt.ylabel('Coefficient Angle (Rad)')
                    plt.savefig(os.path.join(self.figs_dir, 'basis_coeffs.pdf'), format='pdf')
                    # plt.show(block=False)

            sig_filtered_base_real = self.basis_fil_ridge_real[i].predict(sig_bank_mat_combined)
            sig_filtered_base_imag = self.basis_fil_ridge_imag[i].predict(sig_bank_mat_combined)
            self.sig_filtered_base = sig_filtered_base_real + 1j * sig_filtered_base_imag
            self.sig_filtered_base = self.sig_filtered_base.T

            self.sig_filtered_base = self.numpy_transfer(self.sig_filtered_base, dst='context')

            time_delay = self.extract_delay(self.sig_filtered_base, sigs[i, :self.n_samples - shift], self.plot_level >= 5)
            self.print(
                f'Time delay between the signal and its basis filtered version for signal {i + 1}: {time_delay} samples',3)
            # time_delay = 0
            sig_filtered_base_adj, signal_adj, mse, err2sig_ratio = self.time_adjust(self.sig_filtered_base,
                                                                                           sigs[i, :self.n_samples - shift],
                                                                                           time_delay)
            self.print(
                f'Error to signal ratio for the estimation of the main signal using basis filter for signal {i + 1}: {err2sig_ratio}',3)
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
                # plt.show(block=False)


    def basis_filter_param(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        self.print('Beginning to design the optimal basis filter using parameters.',2)


    def visualize_errors(self, mode='ratio'):

        if self.plot_level >= 1:
            self.print('Reporting errors of Wiener and basis filtering.',2)

            if self.basis_errs.shape[0] != self.wiener_errs.shape[0]:
                raise ValueError('Filtering errors size mismatch between wiener and basis filtering {}, {}'.format(self.wiener_errs.shape[0], self.basis_errs.shape[0]))

            plt.figure()
            if mode=='error':
                plt.scatter(np.arange(1, self.N_sig + 1), self.wiener_errs, color='b', label='Wiener')
                plt.scatter(np.arange(1, self.N_sig + 1), self.basis_errs, color='r', label='Basis')
                plt.title('Basis and Wiener errors')
                plt.xlabel('Signal Index')
                plt.ylabel('Error')
            elif mode=='ratio':
                plt.scatter(np.arange(1, self.N_sig + 1), self.basis_errs / self.wiener_errs, color='b', label='B/W')
                plt.scatter(np.arange(1, self.N_sig + 1), self.wiener_errs / self.basis_errs, color='r', label='W/B')
                plt.title('Wiener over basis and basis over wiener errors ratio')
                plt.xlabel('Signal Index')
                plt.ylabel('Ratio')
            plt.legend()
            # plt.show(block=False)

        self.print(f'Mean error to signal ratio for Wiener filtering: {np.mean(self.wiener_errs)}',1)
        self.print(f'Mean error to signal ratio for Basis filtering: {np.mean(self.basis_errs)}',1)


    def visulalize_filter_chars(self):
        if self.plot_level >= 1:
            plt.figure()
            S = np.arange(0, 8).astype(float)
            delay_1 = (3 * 2 ** S - 2) / (2 ** S)
            delay_2 = (2 ** (S + 1) - 1) / (2 ** S)
            plt.plot(S, delay_1, color='b', label='Signal up/down-sampling')
            plt.plot(S, delay_2, color='r', label='Filter up-sampling')
            plt.title('Comaprison of proposed methods and conventional filtering delays')
            plt.xlabel('S (number of stages)')
            plt.ylabel('Normalized delay relative to conventional filtering')
            plt.legend()
            plt.savefig(os.path.join(self.figs_dir, 'filter_delay.pdf'), format='pdf')
            # plt.show()

            plt.figure()
            S = np.arange(1, 8).astype(float)
            opss_1 = 1 / (2 ** (S-2))
            opss_2 = (S + 1) / (2 ** S)
            plt.plot(S, opss_1, color='b', label='Signal up/down-sampling')
            plt.plot(S, opss_2, color='r', label='Filter up-sampling')
            plt.title('Comaprison of proposed methods and conventional \nfiltering needed operations/s')
            plt.xlabel('S (number of stages)')
            plt.ylabel('Normalized ops/s relative to conventional filtering')
            plt.legend()
            plt.savefig(os.path.join(self.figs_dir, 'filter_ops.pdf'), format='pdf')
            # plt.show()


    def visualize_sweeps(self, plot_dic):

        if self.plot_level >= 1:
            colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'purple']

            methods = ['basis', 'wiener']
            fil_orders = [int(i) for i in plot_dic.keys()]
            snrs = [float(i) for i in plot_dic[fil_orders[0]].keys()]
            N_sigs = [int(i) for i in plot_dic[fil_orders[0]][snrs[0]].keys()]
            N_rs = [int(i) for i in plot_dic[fil_orders[0]][snrs[0]][N_sigs[0]].keys()]

            sweep_n_sig = len(N_sigs)>1
            sweep_snr = len(snrs)>1
            sweep_fo = len(fil_orders)>1

            fo_f = fil_orders[self.fo_f_id] if sweep_fo else fil_orders[0]
            snr_f = snrs[self.snr_f_id] if sweep_snr else snrs[0]
            N_sig_f = N_sigs[self.N_sig_f_id] if sweep_n_sig else N_sigs[0]
            N_r_f = N_rs[self.N_r_f_id] if sweep_n_sig else N_rs[0]


            if sweep_n_sig:
                fig, axes = plt.subplots(1, len(N_sigs[1:5]), figsize=(len(N_sigs[1:5])*5, 5), sharey=True)
                for ax, N_sig in zip(axes, N_sigs[1:5]):
                    x = N_rs
                    wiener_err = [plot_dic[fo_f][snr_f][N_sig][N_r]['wiener'] for N_r in N_rs]
                    basis_err = [plot_dic[fo_f][snr_f][N_sig][N_r]['basis'] for N_r in N_rs]
                    ax.semilogx(x, wiener_err, label='Wiener', linestyle='-', marker='o', color='red')
                    ax.semilogx(x, basis_err, label='Basis', linestyle='-', marker='x', color='green')
                    plt.yscale('log')
                    ax.set_title(f'N_sig = {N_sig}')
                    ax.set_xlabel('Number of antennas (Logarithmic)')
                    ax.set_ylabel('Normalized MSE (Logarithmic)')
                    ax.legend()
                    ax.grid(True)
                fig.suptitle('Comparison of Wiener and basis filters error rate for different number of antennas and signals and filter order: {}, SNR: {:0.1f} dB'.format(fo_f, self.lin_to_db(snr_f)))
                plt.savefig(os.path.join(self.figs_dir, 'filter_sw_nr_nsig.pdf'), format='pdf')
                # plt.show()

            if sweep_snr:
                fig, axes = plt.subplots(1, len(fil_orders[0::2]), figsize=(len(fil_orders[0::2])*5, 6), sharey=True)
                # for ax, fo in zip(axes, fil_orders[0::2]):
                for ax, fo in zip([axes], [fo_f]):
                    x = self.lin_to_db(snrs)
                    wiener_err = []
                    basis_err = []
                    for snr in snrs:
                        if self.plot_mean:
                            wiener_err_t=[]
                            basis_err_t=[]
                            for N_r in N_rs:
                                for N_sig in N_sigs:
                                    wiener_err_t.append(plot_dic[fo][snr][N_sig][N_r]['wiener'])
                                    basis_err_t.append(plot_dic[fo][snr][N_sig][N_r]['basis'])
                            wiener_err.append(np.mean(wiener_err_t))
                            basis_err.append(np.mean(basis_err_t))
                        else:
                            wiener_err.append(plot_dic[fo][snr][N_sig_f][N_r_f]['wiener'])
                            basis_err.append(plot_dic[fo][snr][N_sig_f][N_r_f]['basis'])
                    ax.plot(x, wiener_err, label='Wiener', linestyle='-', marker='o', color='red')
                    ax.plot(x, basis_err, label='Basis', linestyle='-', marker='x', color='green')
                    plt.yscale('log')
                    ax.set_title(f'Filter order = {fo}')
                    ax.set_xlabel('SNR (dB)')
                    ax.set_ylabel('Normalized MSE (Logarithmic)')
                    ax.legend()
                    ax.grid(True)
                if self.plot_mean:
                    fig.suptitle('Comparison of Wiener and basis filters error rate for different SNRs and filter orders, averaging on N_sig, N_r')
                else:
                    fig.suptitle('Comparison of Wiener and basis filters error rate for different SNRs and filter orders and N_sig: {}, N_r: {}'.format(N_sig_f, N_r_f))
                plt.savefig(os.path.join(self.figs_dir, 'filter_sw_snr.pdf'), format='pdf')
                # plt.show()

            if sweep_fo:
                fig, axes = plt.subplots(1, len(snrs[0::2]), figsize=(len(snrs[0::2])*5, 6), sharey=True)
                # for ax, snr in zip(axes, snrs[0::2]):
                for ax, snr in zip([axes], [snr_f]):
                    x = fil_orders
                    wiener_err = []
                    basis_err = []
                    for fo in fil_orders:
                        if self.plot_mean:
                            wiener_err_t = []
                            basis_err_t = []
                            for N_r in N_rs:
                                for N_sig in N_sigs:
                                    wiener_err_t.append(plot_dic[fo][snr][N_sig][N_r]['wiener'])
                                    basis_err_t.append(plot_dic[fo][snr][N_sig][N_r]['basis'])
                            wiener_err.append(np.mean(wiener_err_t))
                            basis_err.append(np.mean(basis_err_t))
                        else:
                            wiener_err.append(plot_dic[fo][snr][N_sig_f][N_r_f]['wiener'])
                            basis_err.append(plot_dic[fo][snr][N_sig_f][N_r_f]['basis'])
                    ax.semilogx(x, wiener_err, label='Wiener', linestyle='-', marker='o', color='red')
                    ax.semilogx(x, basis_err, label='Basis', linestyle='-', marker='x', color='green')
                    plt.yscale('log')
                    ax.set_title('SNR = {:0.1f} dB'.format(self.lin_to_db(snr)))
                    ax.set_xlabel('Base filter order (Logarithmic)')
                    ax.set_ylabel('Normalized MSE (Logarithmic)')
                    ax.legend()
                    ax.grid(True)
                if self.plot_mean:
                    fig.suptitle('Comparison of Wiener and basis filters error rate for different filter orders and SNRs, averaging on N_sig, N_r')
                else:
                    fig.suptitle('Comparison of Wiener and basis filters error rate for different filter orders and SNRs and N_sig: {}, N_r: {}'.format(N_sig_f, N_r_f))
                plt.savefig(os.path.join(self.figs_dir, 'filter_sw_fo.pdf'), format='pdf')
                # plt.show()


if __name__ == '__main__':
    pass



from backend import *
from backend import be_np as np, be_scp as scipy



# def use_gpu_context(func):
#     def wrapper(*args, **kwargs):
#         with cp.cuda.Device(0):
#             return func(*args, **kwargs)
#     return wrapper


class filter_utils(object):

    def __init__(self, params):
        self.plot_level = params.plot_level
        self.verbose_level = params.verbose_level
        self.use_gpu = params.use_gpu
        self.gpu_id = params.gpu_id


    def print(self, text='', thr=0):
        if self.verbose_level>=thr:
            print(text)


    def lin_to_db(self, x, mode='pow'):
        if mode=='pow':
            return 10*np.log10(x)
        elif mode=='mag':
            return 20*np.log10(x)

    def db_to_lin(self, x, mode='pow'):
        if mode == 'pow':
            return 10**(x/10)
        elif mode == 'mag':
            return 10**(x/20)


    def upsample(self, signal, up=2):
        """
        Upsample a signal by a factor of 2 by inserting zeros between the original samples.

        Args:
            signal (np.array): Input signal to be upsampled.

        Returns:
            np.array: Upsampled signal with zeros inserted.
        """
        upsampled_length = up * len(signal)
        upsampled_signal = np.zeros(upsampled_length, dtype=complex)

        # Assign the original signal values to the even indices
        upsampled_signal[::up] = signal.copy()

        return upsampled_signal

    def cross_correlation(self, sig_1, sig_2, index):
        if index >= 0:
            padded_sig_2 = np.concatenate((np.zeros(index, dtype=complex), sig_2[:len(sig_2) - index]))
        else:
            padded_sig_2 = np.concatenate((sig_2[-index:], np.zeros(-index, dtype=complex)))

        cros_corr = np.mean(sig_1 * np.conj(padded_sig_2))
        return cros_corr


    def wiener_fir(self, input, output, filter_order_pos, filter_order_neg):

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order + 1
        N_samples = input.shape[1]

        rxx = np.correlate(input[0, :], input[0, :], mode='same') / N_samples
        center = len(rxx) // 2
        rxx = rxx[center:center + filter_length]
        Rxx = scipy.linalg.toeplitz(rxx)

        # Rxx_1 = np.zeros((filter_length, filter_length)).astype(complex)
        # for i in range(filter_length):
        #     for j in range(filter_length):
        #         Rxx_1[i, j] = self.cross_correlation(input[0,:], input[0,:], i - j)
        # print(np.sum(np.abs(Rxx - Rxx_1)))

        ryx = np.correlate(output[0,:], input[0,:], mode='same') / N_samples
        center = len(ryx) // 2
        ryx = ryx[center-filter_order_neg:center+filter_length-filter_order_neg]
        Ryx = np.reshape(ryx, (filter_length, 1))

        # Ryx_1 = np.zeros((filter_length, 1)).astype(complex)
        # for j in range(filter_length):
        #     Ryx_1[j, 0] = self.cross_correlation(output[0,:], input[0,:], j - filter_order_neg)
        # print(np.sum(np.abs(Ryx - Ryx_1)))

        wiener_filter_coef = np.linalg.solve(Rxx, Ryx)

        return wiener_filter_coef


    def wiener_fir_param(self, sig_bw, sig_pwr, sig_cf, spatial_sig, snr, filter_order_pos, filter_order_neg):

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order + 1

        N_in = spatial_sig.shape[0]
        N_out = spatial_sig.shape[1]
        Rxx = np.zeros((filter_length, filter_length)).astype(complex)
        N0 = (np.sum(spatial_sig[0,:]*sig_pwr*sig_bw) / np.sum(sig_bw)) / snr

        for i in range(filter_length):
            for j in range(filter_length):
                t = i-j
                for k in range(N_out):
                    Rxx[i, j] += spatial_sig[0,k] * np.conj(spatial_sig[0,k]) * np.exp(2 * np.pi * 1j * sig_cf[k] * t) * sig_pwr[k] * sig_bw[k] * np.sinc(sig_bw[k]*t)
                if t==0:
                    Rxx[i, j] += N0

        Ryx = np.zeros((filter_length, 1)).astype(complex)
        for j in range(filter_length):
            t = j - filter_order_neg
            for k in range(N_out):
                Ryx[j, 0] = np.conj(spatial_sig[0,k]) * sig_pwr[k] * sig_bw[k] * np.exp(2 * np.pi * 1j * sig_cf[k] * t) * np.sinc(sig_bw[k] * t)

        wiener_filter_coef = np.linalg.solve(Rxx, Ryx)

        return wiener_filter_coef


    def wiener_fir_vector(self, input, output, filter_order_pos, filter_order_neg):
        # input_signals is a N_in x N matrix consisting N_in signals each with N sample points
        # output_signals is N_out x N matrix consisting N_out signals each with N sample points

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order+1
        N_in = input.shape[0]
        N_out = output.shape[0]
        N_samples = input.shape[1]

        # start = time.time()
        Rxx_t=np.zeros((N_in,N_in,filter_length)).astype(complex)
        for i in range(N_in):
            for j in range(i,N_in):
                rxx = np.correlate(input[i,:], input[j,:], mode='same') / N_samples
                center = len(rxx) // 2
                Rxx_t[i,j,:] = rxx[center:center+filter_length]
                Rxx_t[j,i,:] = np.conj(rxx[center:center-filter_length:-1])
        Rxx = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            for j in range(filter_length):
                if j>=i:
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] = Rxx_t[:,:,j-i]
                else:
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] = np.conj(Rxx_t[:,:,i-j]).T

        # Rxx_1 = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        # for i in range(Rxx.shape[0]):
        #     for j in range(Rxx.shape[1]):
        #         idx_1 = i % N_in
        #         idx_2 = j % N_in
        #         corr_index = (j // N_in) - (i // N_in)
        #         Rxx_1[i, j] = self.cross_correlation(input[idx_1, :], input[idx_2, :], corr_index)
        # end=time.time()
        # print('1: {}'.format(end-start))
        # print(np.sum(np.abs(Rxx-Rxx_1)))

        # start = time.time()
        Ryx_t = np.zeros((N_out, N_in, filter_length)).astype(complex)
        for i in range(N_out):
            for j in range(N_in):
                ryx = np.correlate(output[i, :], input[j, :], mode='same') / N_samples
                center = len(ryx) // 2
                Ryx_t[i, j, :] = ryx[center:center + filter_length]
        Ryx = np.zeros((N_out, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            Ryx[:, i * N_in:(i + 1) * N_in] = Ryx_t[:, :, i]

        # Ryx_1 = np.zeros((N_out, filter_length * N_in)).astype(complex)
        # for i in range(Ryx.shape[0]):
        #     for j in range(Ryx.shape[1]):
        #         idx_1 = i
        #         idx_2 = j % N_in
        #         corr_index = j // N_in
        #         Ryx_1[i, j] = self.cross_correlation(output[idx_1, :], input[idx_2, :], corr_index)
        # end = time.time()
        # print('2: {}'.format(end - start))
        # print(np.sum(np.abs(Ryx - Ryx_1)))

        # self.print(f'Rxx determinant: {np.linalg.det(Rxx)}',4)

        # start = time.time()
        wiener_filter_coef = np.linalg.solve(Rxx.T, Ryx.T).T  # Equivalent to Ryx / Rxx
        # end = time.time()
        # print('3: {}'.format(end - start))

        return wiener_filter_coef


    def wiener_fir_vector_param(self, sig_bw, sig_pwr, sig_cf, spatial_sig, snr, filter_order_pos, filter_order_neg):

        # input_signals is a N_in x N matrix consisting N_in signals each with N sample points
        # output_signals is N_out x N matrix consisting N_out signals each with N sample points

        filter_order = filter_order_pos + filter_order_neg
        filter_length = filter_order+1
        N_in = spatial_sig.shape[0]
        N_out = spatial_sig.shape[1]
        N0 = (np.sum(np.mean(spatial_sig,axis=0) * sig_pwr * sig_bw) / np.sum(sig_bw)) / snr

        Rxx = np.zeros((filter_length * N_in, filter_length * N_in)).astype(complex)
        for i in range(filter_length):
            for j in range(filter_length):
                t = j-i
                for k in range(N_out):
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] += np.outer(spatial_sig[:, k], np.conj(spatial_sig[:, k])) * np.exp(2 * np.pi * 1j * sig_cf[k] * t) * sig_pwr[k] * sig_bw[k] * np.sinc(sig_bw[k] * t)
                if t == 0:
                    Rxx[i*N_in:(i+1)*N_in, j*N_in:(j+1)*N_in] += N0 * np.eye(N_in)

        Ryx = np.zeros((N_out, filter_length * N_in)).astype(complex)
        for j in range(filter_length):
            t = j
            temp = None
            for k in range(N_out):
                Rx_a = np.conj(spatial_sig[:, k]) * sig_pwr[k] * sig_bw[k] * np.exp(
                        2 * np.pi * 1j * sig_cf[k] * t) * np.sinc(sig_bw[k] * t)
                Rx_a = Rx_a.reshape(1,-1)
                if temp is None:
                    temp = Rx_a.copy()
                else:
                    temp = np.vstack((temp, Rx_a))
            Ryx[:,j*N_in:(j+1)*N_in] = temp.copy()

        self.print(f'Rxx determinant: {np.linalg.det(Rxx)}',4)

        wiener_filter_coef = np.linalg.solve(Rxx.T, Ryx.T).T  # Equivalent to Ryx / Rxx

        return wiener_filter_coef


    def extract_delay(self, sig_1, sig_2, plot_corr=False):
        """
        Calculate the delay of signal 1 with respect to signal 2 (signal 1 is ahead of signal 2)

        Args:
            sig_1 (np.array): First signal.
            sig_2 (np.array): Second signal.
            plot_corr (bool): Whether to plot the cross-correlation or not.

        Returns:
            delay (int): The delay of signal 1 with respect to signal 2 in samples.
        """
        cross_corr = np.correlate(sig_1, sig_2, mode='full')
        # cross_corr = np.correlate(sig_1, sig_2, mode='same')
        lags = np.arange(-len(sig_2) + 1, len(sig_1))

        if plot_corr:
            plt.figure()
            plt.plot(lags, np.abs(cross_corr), linewidth=1.0)
            plt.title('Cross-Correlation of the two signals')
            plt.xlabel('Lags')
            plt.ylabel('Correlation Coefficient')
            # plt.show()

        max_idx = np.argmax(np.abs(cross_corr))
        delay = int(lags[max_idx])
        # self.print(f'Time delay between the two signals: {delay} samples',4)
        return delay


    def time_adjust(self, sig_1, sig_2, delay):
        """
        Adjust the time of sig_1 with respect to sig_2 based on the given delay.

        Args:
            sig_1 (np.array): First signal.
            sig_2 (np.array): Second signal.
            delay (int): The delay of sig_1 with respect to sig_2 in samples.

        Returns:
            sig_1_adj (np.array): Adjusted sig_1.
            sig_2_adj (np.array): Adjusted sig_2.
            mse (float): Mean squared error between adjusted signals.
            err2sig_ratio (float): Ratio of MSE to mean squared value of sig_2.
        """
        n_points = np.shape(sig_1)[0]

        if delay >= 0:
            sig_1_adj = np.concatenate((sig_1[delay:], np.zeros(delay).astype(complex)))
            sig_2_adj = sig_2.copy()
        else:
            delay = abs(delay)
            sig_1_adj = sig_1.copy()
            sig_2_adj = np.concatenate((sig_2[delay:], np.zeros(delay).astype(complex)))

        mse = float(np.mean(np.abs(sig_1_adj[:n_points-delay] - sig_2_adj[:n_points-delay]) ** 2))
        err2sig_ratio = float(mse / np.mean(np.abs(sig_2) ** 2))

        return sig_1_adj, sig_2_adj, mse, err2sig_ratio



    def basis_fir_us(self, input, fil_base, t, freq, center_freq, iters, us_rate, plot_procedure=False):
        """
        Apply a sequence of basis FIR filters with upsampling to an input signal.

        Args:
            input (np.array): The input signal to the filter.
            fil_base (np.array): Basis filter with which the filtering is being done.
            t (np.array): The time vector.
            freq (np.array): The frequency vector.
            center_freq (float): The center frequency.
            iters (int): Number of iterations of filter upsampling.
            us_rate (int): Upsampling rate.
            plot_procedure (bool): If True, plots the process of DS, filtering, US.

        Returns:
            output (np.array): The output of the filter.
            grp_dly (int): The group delay of the filter.
        """
        om = (freq / max(freq)) * np.pi

        fil_us = [None] * iters
        sig_fil_us = [None] * iters
        fil_base_shifted = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_base)]) * fil_base

        grp_dly = ((len(fil_base)-1) // 2) * (2**(iters + 1) - 1)

        if iters == 0:
            # output = np.convolve(input, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), input)
        else:
            temp = fil_base.copy()
            for i in range(iters):
                # fil_us[i] = upfirdn([1], temp, up=us_rate)
                fil_us[i] = self.upsample(temp, up=us_rate)
                temp = fil_us[i]
            for i in range(iters):
                fil_us[i] = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_us[i])]) * fil_us[i]

            temp = input.copy()
            for i in range(iters):
                # sig_fil_us[i] = np.convolve(temp, fil_us[iters - i - 1], mode='same')
                sig_fil_us[i] = lfilter(fil_us[iters - i - 1], np.array([1]), temp)
                temp = sig_fil_us[i]
            # output = np.convolve(temp, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), temp)

            if plot_procedure:
                # Plotting the filter response and signal spectrums
                plt.figure()
                w, h = freqz(fil_base_shifted, worN=om)
                plt.plot(w/np.pi, self.lin_to_db(abs(h), mode='mag'), linewidth=1.0, label='Base Filter')
                w, h = freqz(fil_us[0], worN=om)
                plt.plot(w/np.pi, self.lin_to_db(abs(h), mode='mag'), linewidth=1.0, label='Upsampled Filter')
                plt.title('Base and Upsampled Filters Frequency Response')
                plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
                plt.ylabel('Magnitude (dB)')
                plt.legend()
                # plt.show()

                plt.figure()
                plot_indices = [0,min(1,iters-1),iters-1]
                # for idx, signal in enumerate(sig_fil_us, start=1):
                for i, idx in enumerate(plot_indices):
                    plt.subplot(len(plot_indices) + 1, 1, i+1)
                    spectrum = np.fft.fftshift(np.fft.fft(sig_fil_us[idx]))
                    plt.plot(freq, self.lin_to_db(np.abs(spectrum), mode='mag'), linewidth=1.0)
                    plt.title(f'Frequency Spectrum of the {idx} round filtered signal')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                plt.subplot(len(plot_indices)+1, 1, len(plot_indices)+1)
                spectrum = np.fft.fftshift(np.fft.fft(output))
                plt.plot(freq, self.lin_to_db(np.abs(spectrum), mode='mag'), linewidth=1.0)
                plt.title('Frequency Spectrum of the output signal')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude (dB)')
                # plt.show()

        # output = np.array(output)
        return output, grp_dly



    def basis_fir_ds_us(self, input, fil_base, t, freq, center_freq, iters, ds_rate, us_rate, plot_procedure=False):
        """
        Apply a basis FIR filter with downsampling and upsampling to an input signal.

        Args:
            input (np.array): The input signal to the filter.
            fil_base (np.array): Basis filter with which the filtering is being done.
            t (np.array): The time vector.
            freq (np.array): The frequency vector.
            center_freq (float): The center frequency.
            iters (int): Number of iterations of downsampling and upsampling.
            ds_rate (int): Downsampling rate.
            us_rate (int): Upsampling rate.
            plot_procedure (bool): If True, plots the process of DS, filtering, US.

        Returns:
            output (np.array): The output of the filter.
            grp_dly (int): The group delay of the filter.
        """
        om = (freq / max(freq)) * np.pi

        sig_ds = [None] * iters
        sig_us = [None] * iters
        sig_ds_fil = [None] * iters
        sig_us_fil = [None] * iters
        fil_base_shifted = np.exp(2 * np.pi * 1j * center_freq * t[:len(fil_base)]) * fil_base
        input_centered = np.exp(-2 * np.pi * 1j * center_freq * t) * input

        grp_dly = ((len(fil_base)-1) // 2) * (3 * (2 ** iters) - 2)
        if iters == 0:
            # output = np.convolve(input, fil_base_shifted, mode='same')
            output = lfilter(fil_base_shifted, np.array([1]), input)
        else:
            temp = input_centered.copy()
            for i in range(iters):
                # sig_ds_fil[i] = np.convolve(temp, fil_base, mode='same')
                sig_ds_fil[i] = lfilter(fil_base, np.array([1]), temp)
                sig_ds[i] = sig_ds_fil[i][::ds_rate]
                temp = sig_ds[i]

            # temp = np.convolve(temp, fil_base, mode='same')
            temp = lfilter(fil_base, np.array([1]), temp)
            for i in range(iters):
                # sig_us[i] = upfirdn([1], temp, up=us_rate)
                sig_us[i] = self.upsample(temp, up=us_rate)
                # sig_us_fil[i] = np.convolve(sig_us[i], fil_base, mode='same')
                sig_us_fil[i] = lfilter(fil_base, np.array([1]), sig_us[i])
                temp = sig_us_fil[i]

            output = sig_us_fil[iters - 1] * (2 ** iters)
            output = np.exp(2 * np.pi * 1j * center_freq * t) * output

            if plot_procedure:
                # Plotting the filter response and signal spectrums
                plt.figure()
                for idx in range(4):
                    plt.subplot(4, 1, idx + 1)
                    if idx == 0:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds_fil[0]))
                        index = 0
                        plt.title('Frequency spectrum of the first round filtered signal')
                    elif idx == 1:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round downsampled filtered signal')
                    elif idx == 2:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds_fil[iters - 1]))
                        index = iters-1
                        plt.title('Frequency spectrum of the last round filtered signal')
                    elif idx == 3:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_ds[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round downsampled filtered signal')
                    spectrum = self.lin_to_db(np.abs(spectrum), mode='mag')
                    freq_ds = freq[::ds_rate**index]
                    plt.plot(freq_ds, spectrum, 'r-', linewidth=1.0)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                plt.figure()
                for idx in range(4):
                    plt.subplot(4, 1, idx + 1)
                    if idx == 0:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round upsampled signal')
                    elif idx == 1:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us_fil[0]))
                        index = 1
                        plt.title('Frequency spectrum of the first round filtered upsampled signal')
                    elif idx == 2:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round upsampled signal')
                    elif idx == 3:
                        spectrum = np.fft.fftshift(np.fft.fft(sig_us_fil[iters - 1]))
                        index = iters
                        plt.title('Frequency spectrum of the last round filtered upsampled signal')
                    spectrum = self.lin_to_db(np.abs(spectrum), mode='mag')
                    freq_us = freq[::us_rate ** (iters - index)]
                    plt.plot(freq_us, spectrum, 'r-', linewidth=1.0)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')

                # plt.show()

        return output, grp_dly



if __name__ == '__main__':
    pass


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz, upfirdn
# from scipy.fftpack import fft, fftshift
from numpy.fft import fft, fftshift
from numpy.random import randn, rand
from filter_utils import *
from sklearn.linear_model import Ridge


if __name__ == '__main__':

    # Constants
    fs = 200e6  # Sampling frequency
    n_points = 2 ** 13
    t = np.arange(0, n_points) / fs  # Time vector

    fil_sharp_bw = 10e6
    fil_base_order_pos = 64
    fil_base_order_neg = 0
    iters = 1
    fil_sharp_order_pos = fil_base_order_pos * (2 ** iters)
    fil_wiener_order_pos = fil_base_order_pos * (2 ** iters)
    fil_wiener_order_neg = 0
    us_rate = 2
    ds_rate = 2

    fil_bank_mode = 1  # 1 for whole-span coverage and 2 for TX signal coverage
    filtering_mode = 1  # 1: use sharp filter bank, 2: use fir_us filters, 3: use fir_ds_us filters
    random_params = True

    if random_params:
        N_sig = 8
        N_r = 8
        # N_sig = 2
        # N_r = 1

        sig_bw = 10e6 + 20e6 * rand(N_sig)
        sig_amp = 1 * np.ones(N_sig) + 4 * rand(N_sig)
        sig_cf = (fs / 2) * (rand(N_sig) - 0.5)
        spatial_sig_rand_coef = 0.9
        spatial_sig = (1 - spatial_sig_rand_coef) * np.ones((N_r, N_sig)) + spatial_sig_rand_coef * rand(N_r, N_sig)
    else:
        N_sig = 8
        N_r = 4
        sig_bw = 1.0e+07 * np.array([2.5152, 1.5262, 1.1372, 2.8934, 2.9045, 2.5694, 2.7378, 1.2596])
        sig_amp = np.array([3.0277, 3.1819, 1.0687, 3.6131, 3.8772, 2.4723, 4.7923, 4.8467])
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
        # sig_amp = np.array([1, 4])
        # sig_cf = np.array([0, 0])
        # spatial_sig = np.array([[1., 1.]])

        # N_sig = 5
        # N_r = 1
        # sig_bw = np.array([60e6, 2e6, 5e6, 6e6, 8e6])
        # sig_amp = np.array([1, 4, 2, 1, 2])
        # sig_cf = np.array([0, 0, 10e6, 20e6, 50e6])
        # spatial_sig = np.array([[1, 1, 1, 1, 1]])


    # Suspicious
    sig_amp = sig_amp.astype(complex)
    spatial_sig = spatial_sig.astype(complex)

    nfft = 2 ** np.ceil(np.log2(n_points)).astype(int)
    snr = 10
    ridge_coeff = 1

    grp_dly_base = (fil_base_order_pos // 2)
    grp_dly_sharp = (fil_sharp_order_pos // 2)

    wiener_errs = np.zeros(N_sig)
    basis_errs = np.zeros(N_sig)

    # print(sig_amp, spatial_sig)
    #=================================================
    om = np.linspace(-np.pi, np.pi, n_points)
    freq = ((np.arange(1, n_points + 1) / n_points) - 0.5) * fs
    #=================================================
    noise = randn(n_points).astype(complex)
    rx = np.zeros((N_r, n_points), dtype=complex)
    signals = np.zeros((N_sig, n_points), dtype=complex)
    sig_sel_id = 0
    rx_sel_id = 0

    for i in range(N_sig):
        fil_sig = firwin(1001, sig_bw[i] / fs)
        # signals[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * sig_amp[i] * np.convolve(noise, fil_sig, mode='same')
        signals[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * sig_amp[i] * lfilter(fil_sig, 1, noise)
        rx += np.outer(spatial_sig[:, i], signals[i, :])

    yvar = np.mean(np.abs(rx) ** 2, axis=1)
    wvar = yvar * (10 ** (-snr / 10))
    # rx += np.sqrt(wvar[:, None] / 2) * noise
    rx += np.outer(np.sqrt(wvar / 2), noise)

    plt.figure()
    plt.subplot(3, 1, 1)
    for i in range(N_sig):
        spectrum = fftshift(fft(signals[i, :]))
        spectrum = 20 * np.log10(np.abs(spectrum))
        plt.plot(freq, spectrum, color=np.random.rand(3), linewidth=0.5)
    plt.title('Frequency spectrum of initial wideband signals')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.subplot(3, 1, 2)
    spectrum = fftshift(fft(rx[rx_sel_id, :]))
    spectrum = 20 * np.log10(np.abs(spectrum))
    plt.plot(freq, spectrum, 'b-', linewidth=0.5)
    plt.title('Frequency spectrum of one of the rx signals')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.subplot(3, 1, 3)
    spectrum = fftshift(fft(signals[sig_sel_id, :]))
    spectrum = 20 * np.log10(np.abs(spectrum))
    plt.plot(freq, spectrum, 'r-', linewidth=0.5)
    plt.title('Frequency spectrum of a selected wideband signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.show()
    # =================================================
    # sig_bw = 10e6
    # iters = 3
    # filter_order = 32
    # fil_sig = firwin(1001, sig_bw / fs)
    # sig_test_1 = lfilter(fil_sig, 1, noise)
    # fil_base = firwin(filter_order + 1, sig_bw*1.5 * (2 ** iters) / fs)
    # fil_sharp = firwin(filter_order*(2 ** iters)+1, sig_bw*1.5 / fs)
    # sig_test_2 = lfilter(fil_sharp, 1, sig_test_1)
    # # sig_test_2, delay = basis_fir_us(input=sig_test_1, fil_base=fil_base, t=t, freq=freq, center_freq=0, iters=iters, us_rate=2, plot_procedure=True)
    # # sig_test_2, delay = basis_fir_ds_us(input=sig_test_1, fil_base=fil_base, t=t, freq=freq, center_freq=0, iters=iters, ds_rate=2, us_rate=2, plot_procedure=True)
    # time_delay = extract_delay(sig_test_2, sig_test_1, True)
    # print(f'Time delay between the signal and its test filtered version for: {time_delay} samples')
    # sig_test_adj, signal_adj, mse, err2sig_ratio = time_adjust(sig_test_2, sig_test_1, time_delay)
    # print(f'Error to signal ratio for the estimation of the test signal using test filter for: {err2sig_ratio}')
    # quit()
    # =================================================
    # Filter bank creation
    if fil_bank_mode == 1:
        fil_bank_num = int(fs / fil_sharp_bw)
        center_freq = (-fs / 2) + (fil_sharp_bw / 2) + np.linspace(0, fil_bank_num - 1, fil_bank_num) * fil_sharp_bw
    elif fil_bank_mode == 2:
        fil_bank_num = N_sig
        center_freq = sig_cf.copy()

    fil_base = [None] * fil_bank_num
    fil_sharp = [None] * fil_bank_num

    for i in range(fil_bank_num):
        if fil_bank_mode == 1:
            fil_bw_base = fil_sharp_bw
        elif fil_bank_mode == 2:
            fil_bw_base = sig_bw[i]
        fil_base[i] = firwin(fil_base_order_pos + 1, fil_bw_base * (2 ** iters) / fs)
        fil_sharp[i] = firwin(fil_sharp_order_pos + 1, fil_bw_base / fs)

    fil_bank = [None] * fil_bank_num
    plt.figure()
    for i in range(fil_bank_num):
        t_fil = t[:len(fil_sharp[i])]
        fil_bank[i] = np.exp(2 * np.pi * 1j * center_freq[i] * t_fil) * fil_sharp[i]
        if i % 1 == 0:
            w, h = freqz(fil_bank[i], worN=om)
            plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), label=f'Filter {i + 1}')
    plt.title('Frequency response of selected filters in the filter bank')
    plt.xlabel('Normalized Frequency (xpi rad/sample)')
    plt.ylabel('Magnitude (dB)')
    # plt.legend()
    plt.show()

    # =================================================
    # Filtering signals
    sig_bank = [[None] * N_r for _ in range(fil_bank_num)]
    for i in range(fil_bank_num):
        for j in range(N_r):
            plot_procedure = (i == int(3 * fil_bank_num / 4) and j == rx_sel_id)
            if filtering_mode==1:
                # sig_bank[i][j] = np.convolve(rx[j, :], fil_bank[i], mode='same')
                sig_bank[i][j] = lfilter(fil_bank[i], 1, rx[j, :])
                filter_delay = grp_dly_sharp
            elif filtering_mode==2:
                sig_bank[i][j], filter_delay = basis_fir_us(rx[j, :], fil_base[i], t, freq, center_freq[i], iters, us_rate, plot_procedure)
            elif filtering_mode == 3:
                sig_bank[i][j], filter_delay = basis_fir_ds_us(rx[j, :], fil_base[i], t, freq, center_freq[i], iters,
                                                               ds_rate, us_rate, plot_procedure)
            else:
                raise ValueError('Invalid Filtering mode %d' % filtering_mode)

            # suspicious
            sig_bank[i][j] = sig_bank[i][j].astype(complex)

    print(f'Total group delay for filtering: {filter_delay}')

    plt.figure()
    for i in range(fil_bank_num):
        if i % 1 == 0:
            spectrum = fftshift(fft(sig_bank[i][rx_sel_id]))
            spectrum = 20 * np.log10(np.abs(spectrum))
            plt.plot(freq, spectrum, color=np.random.rand(3), linewidth=0.5)
    plt.title('Frequency spectrum of the signal bank filtered using filter bank')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.show()
    # =================================================
    # Wiener filtering
    rx_dly = rx
    fil_wiener_single = [[None] * N_r for _ in range(N_sig)]

    if N_r <= 1:
        for i in range(N_sig):
            for j in range(N_r):
                fil_wiener_single[i][j] = wiener_fir(rx, signals[i, :].reshape((1,-1)), fil_wiener_order_pos, fil_wiener_order_neg).reshape(-1)
    else:
        fil_wiener = wiener_fir_vector(rx, signals, fil_wiener_order_pos, fil_wiener_order_neg)
        for i in range(N_sig):
            for j in range(N_r):
                fil_wiener_single[i][j] = fil_wiener[i, j::N_r]

    for i in range(N_sig):
        sig_filtered_wiener = np.zeros_like(t, dtype=complex)
        for j in range(N_r):
            # sig_filtered_wiener += np.convolve(rx_dly[j, :], fil_wiener_single[i][j], mode='same')
            sig_filtered_wiener += lfilter(fil_wiener_single[i][j], 1, rx_dly[j, :])

        time_delay = extract_delay(sig_filtered_wiener, signals[i, :], False)
        print(f'Time delay between the signal and its Wiener filtered version for {i + 1}: {time_delay} samples')

        sig_filtered_wiener_adj, signal_adj, mse, err2sig_ratio = time_adjust(sig_filtered_wiener, signals[i, :],
                                                                              time_delay)
        print(
            f'Error to signal ratio for the estimation of the main signal using Wiener filter for {i + 1}: {err2sig_ratio}')
        wiener_errs[i] = err2sig_ratio

        if i == sig_sel_id:
            plt.figure()
            index = range(n_points // 2,n_points // 2+500)
            plt.plot(t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
            plt.plot(t[index], np.abs(sig_filtered_wiener_adj[index]), 'b-', linewidth=0.5)
            plt.title('Signal and its recovered wiener filtered in time domain')
            plt.xlabel('Time(s)')
            plt.ylabel('Magnitude')
            plt.show()

    plt.figure()
    w, h = freqz(fil_wiener_single[sig_sel_id][rx_sel_id], worN=om)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
    plt.title('Frequency response of the Wiener filter for the selected TX signal and RX antenna')
    plt.xlabel('Normalized Frequency (xpi rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.show()
    # =================================================
    # Basis filtering
    shift = filter_delay
    sig_bank_mat = np.zeros((n_points - shift, fil_bank_num * N_r), dtype=complex)
    for j in range(N_r):
        for i in range(fil_bank_num):
            sig_bank_mat[:, (j * fil_bank_num + i)] = sig_bank[i][j][shift:]
    b = np.copy(signals[:, :n_points -shift].T)

    for i in range(N_sig):
        # # sig_bank_coeffs = np.linalg.lstsq(sig_bank_mat.T @ sig_bank_mat + ridge_coeff * np.eye(fil_bank_num * N_r), sig_bank_mat.T @ b[:,i],
        # #                 rcond=None)[0]
        # sig_bank_coeffs = np.linalg.inv(sig_bank_mat.T @ sig_bank_mat + (ridge_coeff * np.eye(fil_bank_num * N_r))) @ (sig_bank_mat.T) @ b[:,i]
        # sig_filtered_base = (sig_bank_mat @ sig_bank_coeffs).T

        sig_bank_mat_real = np.real(sig_bank_mat)
        sig_bank_mat_imag = np.imag(sig_bank_mat)
        sig_bank_mat_combined = np.hstack([sig_bank_mat_real, sig_bank_mat_imag])
        b_real = np.real(b[:,i])
        b_imag = np.imag(b[:,i])
        ridge_real = Ridge(alpha=1.0)
        ridge_imag = Ridge(alpha=1.0)
        ridge_real.fit(sig_bank_mat_combined, b_real)
        ridge_imag.fit(sig_bank_mat_combined, b_imag)

        sig_filtered_base_real = ridge_real.predict(sig_bank_mat_combined)
        sig_filtered_base_imag = ridge_imag.predict(sig_bank_mat_combined)
        sig_filtered_base = sig_filtered_base_real + 1j*sig_filtered_base_imag
        sig_filtered_base = sig_filtered_base.T

        sig_bank_coeffs_real = ridge_real.coef_
        sig_bank_coeffs_imag = ridge_imag.coef_
        sig_bank_coeffs_real_real = sig_bank_coeffs_real[:sig_bank_mat.shape[1]]
        sig_bank_coeffs_real_imag = sig_bank_coeffs_real[sig_bank_mat.shape[1]:]
        sig_bank_coeffs_imag_real = sig_bank_coeffs_imag[:sig_bank_mat.shape[1]]
        sig_bank_coeffs_imag_imag = sig_bank_coeffs_imag[sig_bank_mat.shape[1]:]

        sig_bank_multiplied = np.multiply(sig_bank_coeffs_real_real.reshape((1,-1)), sig_bank_mat_real) \
                            + np.multiply(sig_bank_coeffs_real_imag.reshape((1,-1)), sig_bank_mat_imag) \
                            + np.multiply(sig_bank_coeffs_imag_real.reshape((1,-1)), sig_bank_mat_real*1j) \
                            + np.multiply(sig_bank_coeffs_imag_imag.reshape((1,-1)), sig_bank_mat_imag*1j)
        sig_bank_coeffs_mat = np.divide(sig_bank_multiplied, sig_bank_mat)
        sig_bank_coeffs = np.mean(sig_bank_coeffs_mat, axis=0).reshape(-1)
        # var_mat = (sig_bank_coeffs_mat-np.tile(sig_bank_coeffs, (sig_bank_coeffs_mat.shape[0],1)))**2
        # print(np.mean(var_mat, axis=0))



        time_delay = extract_delay(sig_filtered_base, signals[i, :n_points-shift], False)
        print(f'Time delay between the signal and its basis filtered version for {i + 1}: {time_delay} samples')
        # time_delay = 0
        sig_filtered_base_adj, signal_adj, mse, err2sig_ratio = time_adjust(sig_filtered_base,
                                                                            signals[i, :n_points-shift], time_delay)
        print(
            f'Error to signal ratio for the estimation of the main signal using basis filter for {i + 1}: {err2sig_ratio}')
        basis_errs[i] = err2sig_ratio

        if i == sig_sel_id:
            plt.figure()
            index = range(n_points // 2,n_points // 2+500)
            plt.plot(t[index], np.abs(signal_adj[index]),'r-', linewidth=0.5)
            # plt.plot(t[index], np.abs(signals[i, index]), 'b-', linewidth=0.5)
            plt.plot(t[index], np.abs(sig_filtered_base_adj[index]), 'b-', linewidth=0.5)
            # plt.plot(t[index], np.abs(sig_filtered_base[i, index]),'r-', linewidth=0.5)
            plt.title('Signal and its recovered basis filtered in time domain')
            plt.xlabel('Time(s)')
            plt.ylabel('Magnitude')
            plt.show()

            freq_range = center_freq
            coeffs_range = np.arange(rx_sel_id * fil_bank_num, rx_sel_id * fil_bank_num + fil_bank_num)
            coeffs = np.abs(sig_bank_coeffs[coeffs_range])

            if fil_bank_mode == 2:
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
    # =================================================
    # plt.figure()
    # plt.scatter(np.arange(1, N_sig + 1), wiener_errs, color='b', label='Wiener')
    # plt.scatter(np.arange(1, N_sig + 1), basis_errs, color='r', label='Basis')
    # plt.legend()
    # plt.title('Basis and Wiener errors')
    # plt.xlabel('Signal Index')
    # plt.ylabel('Error')
    # plt.show()

    plt.figure()
    plt.scatter(np.arange(1, N_sig + 1), basis_errs / wiener_errs, color='b', label='B/W')
    plt.scatter(np.arange(1, N_sig + 1), wiener_errs / basis_errs, color='r', label='W/B')
    plt.legend()
    plt.title('Wiener over basis and basis over wiener errors ratio')
    plt.xlabel('Signal Index')
    plt.ylabel('Ratio')
    plt.show()

    print(f'Mean error to signal ratio for Wiener filtering: {np.mean(wiener_errs)}')
    print(f'Mean error to signal ratio for Basis filtering: {np.mean(basis_errs)}')

    # =================================================


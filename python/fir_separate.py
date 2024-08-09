from backend import *
from backend import be_np as np, be_scp as scipy
from filter_utils import filter_utils



class fir_separate(object):
    def __init__(self, params):

        self.n_samples = params.n_samples
        self.fs = params.fs
        self.N_r = params.N_r
        self.N_sig = params.N_sig
        self.sharp_bw = params.sharp_bw
        self.base_order_pos = params.base_order_pos
        self.base_order_neg = params.base_order_neg
        self.n_stage = params.n_stage
        self.us_rate = params.us_rate
        self.ds_rate = params.ds_rate
        self.fil_bank_mode = params.fil_bank_mode
        self.fil_mode = params.fil_mode
        self.snr = params.snr
        self.cf_range = params.cf_range
        self.bw_range = params.bw_range
        self.psd_range = params.psd_range
        self.sig_noise = params.sig_noise
        self.ridge_coeff = params.ridge_coeff
        self.sig_sel_id = params.sig_sel_id
        self.rx_sel_id = params.rx_sel_id
        self.spat_sig_range = params.spat_sig_range
        self.plot_level = params.plot_level
        self.verbose_level = params.verbose_level
        self.rand_params = params.rand_params
        self.use_gpu = params.use_gpu
        self.gpu_id = params.gpu_id
        self.figs_dir = params.figs_dir

        self.sharp_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.sharp_order_neg = self.base_order_neg * (2 ** self.n_stage)
        self.wiener_order_pos = self.base_order_pos * (2 ** self.n_stage)
        self.wiener_order_neg = self.base_order_neg * (2 ** self.n_stage)

        self.grp_dly_base = (self.base_order_pos // 2)
        self.grp_dly_sharp = (self.sharp_order_pos // 2)

        self.t = np.arange(0, self.n_samples) / self.fs  # Time vector
        self.freq = ((np.arange(1, self.n_samples + 1) / self.n_samples) - 0.5) * self.fs
        self.om = np.linspace(-np.pi, np.pi, self.n_samples)
        self.nfft = 2 ** np.ceil(np.log2(self.n_samples)).astype(int)

        self.futil = filter_utils(params=params)

        self.basis_fil_ridge_real = None
        self.basis_fil_ridge_imag = None

        self.print('Initialized the fir_separate class instance.',2)


    def print(self, text='', thr=0):
        if self.verbose_level>=thr:
            print(text)


    def check_cupy_gpu(self, gpu_id=0):
        if not import_cupy:
            return False
        try:
            import cupy as cp
            # Check if CuPy is installed
            print("CuPy version: {}".format(cp.__version__))

            num_gpus = cp.cuda.runtime.getDeviceCount()
            print(f"Number of GPUs available: {num_gpus}")

            # Check if the GPU is available
            cp.cuda.Device(gpu_id).compute_capability
            print("GPU {} is available".format(gpu_id))

            print('GPU {} properties: {}'.format(gpu_id, cp.cuda.runtime.getDeviceProperties(gpu_id)))
            return True
        except ImportError:
            print("CuPy is not installed.")
        except:
            print("GPU is not available or CUDA is not installed correctly.")
        return False


    def get_gpu_device(self):
        if self.use_gpu:
            return np.cuda.Device(self.gpu_id)
        else:
            return None


    def check_gpu_usage(self):
        with np.cuda.Device(self.gpu_id) as device:
            self.print(f"Current device: {device}",0)


    def print_gpu_memory(self):
        with np.cuda.Device(self.gpu_id):
            mempool = np.get_default_memory_pool()
            self.print("Used GPU memory: {} bytes".format(mempool.used_bytes()),0)
            self.print("Total GPU memory: {} bytes".format(mempool.total_bytes()),0)


    # Initialize and warm-up
    def warm_up_gpu(self):
        self.print('Starting GPU warmup.', 0)
        with np.cuda.Device(self.gpu_id):
            start = time.time()
            _ = np.array([1, 2, 3])
            _ = np.array([4, 5, 6])
            a = np.random.rand(1000, 1000)
            _ = np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
            _ = np.dot(a, a)
            np.cuda.Stream.null.synchronize()
            end = time.time()
        self.print("GPU warmup time: {}".format(end-start),0)


    # Perform computation
    def gpu_cpu_compare(self, size=20000):
        self.print('Starting CPU and GPU times compare.', 0)
        # Generate data
        a_cpu = numpy.random.rand(size, size).astype(np.float32)
        b_cpu = numpy.random.rand(size, size).astype(np.float32)

        # Measure CPU time for comparison
        start = time.time()
        result_cpu = numpy.dot(a_cpu, b_cpu)
        end = time.time()
        cpu_time = end - start
        self.print("CPU time: {}".format(cpu_time),0)

        with np.cuda.Device(self.gpu_id):
            # Transfer data to GPU
            a_gpu = np.asarray(a_cpu)
            b_gpu = np.asarray(b_cpu)
            # a_gpu = cp.random.rand(size, size).astype(cp.float32)
            # b_gpu = cp.random.rand(size, size).astype(cp.float32)

            # Measure GPU time
            start = time.time()
            result_gpu = np.dot(a_gpu, b_gpu)
            np.cuda.Stream.null.synchronize()  # Ensure all computations are finished
            end = time.time()
            gpu_time = end - start
            self.print("GPU time: {}".format(gpu_time), 0)

        return gpu_time, cpu_time


    def numpy_transfer(self, arrays, dst='numpy'):

        if import_cupy:
            if isinstance(arrays, list):
                out = []
                for i in range(len(arrays)):
                    if dst == 'numpy' and not isinstance(arrays[i], numpy.ndarray):
                        out.append(np.asnumpy(arrays[i]))
                    elif dst == 'context' and not isinstance(arrays[i], np.ndarray):
                        out.append(np.asarray(arrays[i]))
            else:
                if dst=='numpy' and not isinstance(arrays, numpy.ndarray):
                    out = np.asnumpy(arrays)
                elif dst=='context' and not isinstance(arrays, np.ndarray):
                    out = np.asarray(arrays)
        else:
            out = arrays
        return out


    def gen_rand_params(self):
        self.print('Generating a set of random parameters.', 2)

        if self.rand_params:
            sig_bw = uniform(self.bw_range[0], self.bw_range[1], self.N_sig)
            psd_range = self.psd_range/1e3/1e6
            sig_psd = uniform(psd_range[0], psd_range[1], self.N_sig)
            sig_cf = uniform(self.cf_range[0], self.cf_range[1], self.N_sig)
            # spatial_sig = uniform(self.spat_sig_range[0], self.spat_sig_range[1], (self.N_r, self.N_sig))
            spat_sig_mag = uniform(self.spat_sig_range[0], self.spat_sig_range[1], (self.N_r, self.N_sig))
            spat_sig_ang = uniform(0, 2 * np.pi, (self.N_r, self.N_sig))
            spatial_sig = spat_sig_mag * np.cos(spat_sig_ang) + 1j * spat_sig_mag * np.sin(spat_sig_ang)
        else:
            self.N_sig = 8
            self.N_r = 4
            sig_bw = np.array([22323529.99278623, 48200108.09963198, 36547482.82140353, 12584112.2088698,
                               15527035.64753453, 46191498.35367701, 34587812.69753684, 38755537.21014])
            sig_psd = np.array([1.23803258e-10 + 0.j, 1.90150973e-10 + 0.j, 1.30873743e-10 + 0.j,
                                1.39155310e-10 + 0.j, 1.11357556e-10 + 0.j, 2.49922447e-10 + 0.j,
                                1.95276336e-10 + 0.j, 9.26415974e-11 + 0.j])
            sig_cf = np.array([ -9339579.67553654,  -7771172.84426754,  -3357446.53363527,
                                24684146.71413184, -33177215.5784691,   34935824.59324883,
                                -42989233.3315676,   28144876.63742255])
            spatial_sig = np.array([[ 0.73816382+0.09205707j,  0.70387533+0.15183863j,  0.54803076-0.19402354j,
                                      -0.58491791+0.5058128j,  -0.16193047+0.56388011j,  0.51477254-0.54550331j,
                                      -0.33295462-0.64565756j, -0.03176378-0.47204342j],
                                    [-0.04677747+0.13017887j,  0.63423703+0.13803569j, 0.25751322-0.62836306j,
                                     -0.57574726-0.73478361j, -0.0835737 +0.14649245j,  0.87551691-0.10826231j,
                                     -0.62712629+0.50051845j,  0.24537359-0.27019111j],
                                    [ 0.04467297-0.18440435j, -0.06279467-0.54858543j, -0.04531414-0.19584594j,
                                      0.13795275+0.1346313j,  -0.14161517-0.1081777j,  -0.06453969-0.09272613j,
                                      0.47966132+0.2059196j,  -0.62141369+0.48982126j],
                                    [ 0.21949995-0.67689595j,  0.68860328-0.56853604j,  0.62686718+0.25288625j,
                                      -0.44005474-0.18483065j,  0.30352992+0.9416349j,   0.49304309+0.47657541j,
                                      -0.72656496+0.09926177j,  0.23935423+0.35924608j]])

            # self.N_sig = 2
            # self.N_r = 1
            # sig_bw = np.array([60e6, 2e6])
            # sig_psd = (1/self.fs) * np.array([1, 4])
            # sig_cf = np.array([0, 0])
            # spatial_sig = np.array([[1., 1.]])

            # self.N_sig = 5
            # self.N_r = 1
            # sig_bw = np.array([40e6, 2e6, 5e6, 6e6, 8e6])
            # sig_psd = (1/self.fs) * np.array([1, 4, 2, 1, 2])
            # sig_cf = np.array([1e6, 1e6, 10e6, 20e6, 50e6])
            # spatial_sig = np.array([[1, 1, 1, 1, 1]])

        sig_psd = sig_psd.astype(complex)
        spatial_sig = spatial_sig.astype(complex)

        return (sig_bw, sig_psd, sig_cf, spatial_sig)


    def gen_noise(self, mode='complex'):
        if mode=='real':
            noise = randn(self.n_samples).astype(complex)           # Generate noise with PSD=1/fs W/Hz
            # noise = normal(loc=0, scale=1, size=self.n_samples).astype(complex)
        elif mode=='complex':
            noise = (randn(self.n_samples) + 1j*randn(self.n_samples)).astype(complex)           # Generate noise with PSD=2/fs W/Hz

        return noise


    def generate_signals(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        self.print('Generating a set of signals and a rx signal.',2)

        rx = np.zeros((self.N_r, self.n_samples), dtype=complex)
        sigs = np.zeros((self.N_sig, self.n_samples), dtype=complex)

        for i in range(self.N_sig):
            fil_sig = firwin(1001, sig_bw[i] / self.fs)
            # sigs[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * t) * sig_psd[i] * np.convolve(noise, fil_sig, mode='same')
            sigs[i, :] = np.exp(2 * np.pi * 1j * sig_cf[i] * self.t) * np.sqrt(
                sig_psd[i]*(self.fs/2)) * lfilter(fil_sig, np.array([1]), self.gen_noise(mode='complex'))
            rx += np.outer(spatial_sig[:, i], sigs[i, :])

            if self.sig_noise:
                yvar = np.mean(np.abs(sigs[i, :]) ** 2)
                wvar = yvar / self.snr
                sigs[i, :] += np.sqrt(wvar / 2) * self.gen_noise(mode='complex')

        yvar = np.mean(np.abs(rx) ** 2, axis=1)
        wvar = yvar / self.snr
        noise_rx = np.array([self.gen_noise(mode='complex') for _ in range(self.N_r)])
        # rx += np.sqrt(wvar[:, None] / 2) * noise
        # rx += np.outer(np.sqrt(wvar / 2), self.gen_noise(mode='complex'))
        rx += np.sqrt(wvar[:, None] / 2) * noise_rx

        if self.plot_level >= 2:
            plt.figure()
            # plt.figure(figsize=(10,6))
            # plt.tight_layout()
            plt.subplots_adjust(wspace=0.5, hspace=1.0)
            plt.subplot(3, 1, 1)
            for i in range(self.N_sig):
                spectrum = fftshift(fft(sigs[i, :]))
                spectrum = 20 * np.log10(np.abs(spectrum))
                plt.plot(self.freq, spectrum, color=rand(3), linewidth=0.5)
            plt.title('Frequency spectrum of initial wideband signals')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')

            plt.subplot(3, 1, 2)
            spectrum = fftshift(fft(rx[self.rx_sel_id, :]))
            spectrum = 20 * np.log10(np.abs(spectrum))
            plt.plot(self.freq, spectrum, 'b-', linewidth=0.5)
            plt.title('Frequency spectrum of RX signal in a selected antenna')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')

            plt.subplot(3, 1, 3)
            spectrum = fftshift(fft(sigs[self.sig_sel_id, :]))
            spectrum = 20 * np.log10(np.abs(spectrum))
            plt.plot(self.freq, spectrum, 'r-', linewidth=0.5)
            plt.title('Frequency spectrum of the desired wideband signal')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')

            plt.savefig(self.figs_dir + 'tx_rx_sigs.pdf', format='pdf')
            # plt.show(block=False)

        # frequencies, psd = welch(self.gen_noise(mode='complex'), self.fs, nperseg=1024)
        # frequencies, psd = welch(sigs[0,:], self.fs, nperseg=1024)
        # frequencies, psd = welch(rx[0,:], self.fs, nperseg=1024)
        # plt.figure(figsize=(10, 6))
        # plt.semilogy(frequencies, psd)
        # plt.title('Power Spectral Density (PSD) of Signal')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel(r'PSD ($V^2$/Hz)')
        # plt.grid(True)
        # plt.show()
        # raise InterruptedError("Plot interrupt")

        return (rx, sigs)


    def wiener_filter_design(self, rx, sigs):
        self.print('Beginning to design the optimal wiener filter using the rx and desired signals.',2)

        # N_sig = sigs.shape[0]
        # N_r = rx.shape[0]
        # n_samples = sigs.shape[1]

        self.fil_wiener_single = [[None] * self.N_r for _ in range(self.N_sig)]

        if self.N_r <= 1:
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = self.futil.wiener_fir(rx, sigs[i, :].reshape((1, -1)), self.wiener_order_pos,
                                                                         self.wiener_order_neg).reshape(-1)
        else:
            fil_wiener = self.futil.wiener_fir_vector(rx, sigs, self.wiener_order_pos, self.wiener_order_neg)
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::self.N_r]

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
            for rx_id in range(self.N_r):
                plt.subplot(self.N_r,1,rx_id+1)
                w, h = freqz(self.fil_wiener_single[self.sig_sel_id][rx_id], worN=self.om)
                plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
                plt.title('Selected TX signal, and RX antenna {}'.format(rx_id+1))
                if rx_id == 1:
                    plt.ylabel('Magnitude (dB)')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.savefig(self.figs_dir + 'wiener_filters.pdf', format='pdf')
            # plt.show(block=False)

        return self.fil_wiener_single


    def wiener_filter_apply(self, rx, sigs):
        self.print('Beginning to apply the optimal wiener filter on the rx signal.',2)

        rx_dly = rx.copy()
        self.wiener_errs = np.zeros(self.N_sig)

        for i in range(self.N_sig):
            sig_filtered_wiener = np.zeros_like(self.t, dtype=complex)
            for j in range(self.N_r):
                # sig_filtered_wiener += np.convolve(rx_dly[j, :], fil_wiener_single[i][j], mode='same')
                sig_filtered_wiener += lfilter(self.fil_wiener_single[i][j], np.array([1]), rx_dly[j, :])

            time_delay = self.futil.extract_delay(sig_filtered_wiener, sigs[i, :], self.plot_level >= 5)
            self.print(f'Time delay between the signal and its Wiener filtered version for signal {i + 1}: {time_delay} samples',3)

            sig_filtered_wiener_adj, signal_adj, mse, err2sig_ratio = self.futil.time_adjust(sig_filtered_wiener, sigs[i, :],
                                                                                             time_delay)
            self.print(
                f'Error to signal ratio for the estimation of the main signal using Wiener filter for signal {i + 1}: {err2sig_ratio}',3)
            self.wiener_errs[i] = err2sig_ratio

            if i == self.sig_sel_id and self.plot_level >= 4:
                plt.figure()
                index = range(self.n_samples // 2, self.n_samples // 2 + 500)
                plt.plot(self.t[index], np.abs(signal_adj[index]), 'r-', linewidth=0.5)
                plt.plot(self.t[index], np.abs(sig_filtered_wiener_adj[index]), 'b-', linewidth=0.5)
                plt.title('Signal and its recovered wiener filtered in time domain')
                plt.xlabel('Time(s)')
                plt.ylabel('Magnitude')
                # plt.show(block=False)


    def wiener_filter_param(self, sig_bw, sig_psd, sig_cf, spatial_sig):
        self.print('Beginning to design the optimal wiener filter using parameters.',2)

        # N_sig = spatial_sig.shape[1]
        # N_r = spatial_sig.shape[0]

        self.wiener_errs_param = np.zeros(self.N_sig)
        self.fil_wiener_single = [[None] * self.N_r for _ in range(self.N_sig)]

        if self.N_r <= 1:
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = self.futil.wiener_fir_param(sig_bw, sig_psd, sig_cf, spatial_sig, self.snr, self.wiener_order_pos,
                                                                               self.wiener_order_neg).reshape(-1)
        else:
            fil_wiener = self.futil.wiener_fir_vector_param(sig_bw, sig_psd, sig_cf, spatial_sig, self.snr, self.wiener_order_pos, self.wiener_order_neg)
            for i in range(self.N_sig):
                for j in range(self.N_r):
                    self.fil_wiener_single[i][j] = fil_wiener[i, j::self.N_r]

        if self.plot_level >= 3:
            plt.figure()
            w, h = freqz(self.fil_wiener_single[self.sig_sel_id][self.rx_sel_id], worN=self.om)
            plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), linewidth=1.0)
            plt.title('Frequency response of the parametric Wiener filter \n for the selected TX signal and RX antenna')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.ylabel('Magnitude (dB)')
            # plt.show(block=False)


    def basis_filter_design(self, rx, sigs, sig_bw, sig_cf):
        self.print('Beginning to design the optimal basis filters using the rx and desired signals data.',2)

        # N_sig = sigs.shape[0]
        # N_r = rx.shape[0]
        # n_samples = sigs.shape[1]

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
                    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), label=f'Filter {i + 1}')
            plt.title('Frequency response of basis filters in the filter bank')
            plt.xlabel(r'Normalized Frequency ($\times \pi$ rad/sample)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(self.figs_dir + 'basis_filters.pdf', format='pdf')
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
                    self.sig_bank[i][j], self.filter_delay = self.futil.basis_fir_us(rx[j, :], self.fil_base[i], self.t, self.freq,
                                                                                     self.fil_cf[i], self.n_stage,
                                                                                     self.us_rate, plot_procedure)
                elif self.fil_mode == 3:
                    self.sig_bank[i][j], self.filter_delay = self.futil.basis_fir_ds_us(rx[j, :], self.fil_base[i], self.t,
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
                    spectrum = 20 * np.log10(np.abs(spectrum))
                    plt.plot(self.freq, spectrum, color=rand(3), linewidth=0.5)
            plt.title('Frequency spectrum of the signal bank filtered using the filter bank')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.savefig(self.figs_dir + 'signal_bank.pdf', format='pdf')
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
                    plt.savefig(self.figs_dir + 'basis_coeffs.pdf', format='pdf')
                    # plt.show(block=False)

            sig_filtered_base_real = self.basis_fil_ridge_real[i].predict(sig_bank_mat_combined)
            sig_filtered_base_imag = self.basis_fil_ridge_imag[i].predict(sig_bank_mat_combined)
            self.sig_filtered_base = sig_filtered_base_real + 1j * sig_filtered_base_imag
            self.sig_filtered_base = self.sig_filtered_base.T

            self.sig_filtered_base = self.numpy_transfer(self.sig_filtered_base, dst='context')

            time_delay = self.futil.extract_delay(self.sig_filtered_base, sigs[i, :self.n_samples - shift], self.plot_level >= 5)
            self.print(
                f'Time delay between the signal and its basis filtered version for signal {i + 1}: {time_delay} samples',3)
            # time_delay = 0
            sig_filtered_base_adj, signal_adj, mse, err2sig_ratio = self.futil.time_adjust(self.sig_filtered_base,
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
            plt.savefig(self.figs_dir + 'filter_delay.pdf', format='pdf')
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
            plt.savefig(self.figs_dir + 'filter_ops.pdf', format='pdf')
            # plt.show()


    def plot(self, plot_dic):

        if self.plot_level >= 1:
            colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'purple']

            fil_orders = [int(i) for i in plot_dic.keys()]
            snrs = [float(i) for i in plot_dic[fil_orders[0]].keys()]
            N_sigs = [int(i) for i in plot_dic[fil_orders[0]][snrs[0]].keys()]
            N_rs = [int(i) for i in plot_dic[fil_orders[0]][snrs[0]][N_sigs[0]].keys()]
            methods = ['basis', 'wiener']
            fo_f = fil_orders[5] if len(fil_orders)>5 else fil_orders[0]
            snr_f = snrs[5] if len(snrs)>5 else snrs[0]
            N_sig_f = N_sigs[3] if len(N_sigs)>3 else N_sigs[0]
            N_r_f = N_rs[3] if len(N_rs)>3 else N_rs[0]

            sweep_n_sig = len(N_sigs)>1
            sweep_snr = len(snrs)>1
            sweep_fo = len(fil_orders)>1

            if sweep_n_sig:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
                for ax, N_sig in zip(axes, N_sigs[2:5]):
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
                fig.suptitle('Comparison of Wiener and basis filters error rate for different number of antennas and signals and filter order: {}, SNR: {}'.format(fo_f, snr_f))
                plt.savefig(self.figs_dir + '{}.pdf'.format('filter_sw_nr_nsig'), format='pdf')
                # plt.show()

            if sweep_snr:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
                # for ax, fo in zip(axes, fil_orders[3:6]):
                for ax, fo in zip([axes], [fo_f]):
                    x = 10*np.log10(snrs)
                    wiener_err = []
                    basis_err = []
                    for snr in snrs:
                        wiener_err_t=[]
                        basis_err_t=[]
                        for N_r in N_rs:
                            for N_sig in N_sigs:
                                wiener_err_t.append(plot_dic[fo][snr][N_sig][N_r]['wiener'])
                                basis_err_t.append(plot_dic[fo][snr][N_sig][N_r]['basis'])
                        wiener_err.append(np.mean(wiener_err_t))
                        basis_err.append(np.mean(basis_err_t))
                    # wiener_err = [plot_dic[fo][snr][N_sig_f][N_r_f]['wiener'] for snr in snrs]
                    # basis_err = [plot_dic[fo][snr][N_sig_f][N_r_f]['basis'] for snr in snrs]
                    ax.plot(x, wiener_err, label='Wiener', linestyle='-', marker='o', color='red')
                    ax.plot(x, basis_err, label='Basis', linestyle='-', marker='x', color='green')
                    plt.yscale('log')
                    ax.set_title(f'Filter order = {fo}')
                    ax.set_xlabel('SNR (dB)')
                    ax.set_ylabel('Normalized MSE (Logarithmic)')
                    ax.legend()
                    ax.grid(True)
                fig.suptitle(
                    'Comparison of Wiener and basis filters error rate for different SNRs and filter orders and N_sig: {}, N_r: {}'.format(
                        N_sig_f, N_r_f))
                plt.savefig(self.figs_dir + '{}.pdf'.format('filter_sw_snr'), format='pdf')
                # plt.show()

            if sweep_fo:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
                # for ax, snr in zip(axes, snrs[3:7]):
                for ax, snr in zip([axes], [snr_f]):
                    x = fil_orders
                    wiener_err = []
                    basis_err = []
                    for fo in fil_orders:
                        wiener_err_t = []
                        basis_err_t = []
                        for N_r in N_rs:
                            for N_sig in N_sigs:
                                wiener_err_t.append(plot_dic[fo][snr][N_sig][N_r]['wiener'])
                                basis_err_t.append(plot_dic[fo][snr][N_sig][N_r]['basis'])
                        wiener_err.append(np.mean(wiener_err_t))
                        basis_err.append(np.mean(basis_err_t))
                    # wiener_err = [plot_dic[fo][snr][N_sig_f][N_r_f]['wiener'] for fo in fil_orders]
                    # basis_err = [plot_dic[fo][snr][N_sig_f][N_r_f]['basis'] for fo in fil_orders]
                    ax.semilogx(x, wiener_err, label='Wiener', linestyle='-', marker='o', color='red')
                    ax.semilogx(x, basis_err, label='Basis', linestyle='-', marker='x', color='green')
                    plt.yscale('log')
                    ax.set_title('SNR = {:0.3f}'.format(snr))
                    ax.set_xlabel('Base filter order (Logarithmic)')
                    ax.set_ylabel('Normalized MSE (Logarithmic)')
                    ax.legend()
                    ax.grid(True)
                fig.suptitle(
                    'Comparison of Wiener and basis filters error rate for different filter orders and SNRs and N_sig: {}, N_r: {}'.format(
                        N_sig_f, N_r_f))
                plt.savefig(self.figs_dir + '{}.pdf'.format('filter_sw_fo'), format='pdf')
                # plt.show()



if __name__ == '__main__':
    pass


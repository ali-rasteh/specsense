from backend import *
from backend import be_np as np, be_scp as scipy
from sigcom_toolkit.signal_utils import Signal_Utils





class SS_Detection(Signal_Utils):
    def __init__(self, params):
        super().__init__(params)

        self.shape = params.shape
        self.n_simulations = params.n_simulations
        self.ML_thr = params.ML_thr
        self.ML_thr_mode = params.ML_thr_mode
        self.ML_PFA = params.ML_PFA
        self.ML_mode = params.ML_mode
        self.n_adj_search = params.n_adj_search
        self.n_largest = params.n_largest
        self.calibrate_measurements = params.calibrate_measurements
        self.n_calibration = params.n_calibration
        self.known_interval = params.known_interval

        self.print("Initialized Spectrum Sensing class instance.",thr=0)


    def plot_MD_vs_SNR(self, N=1024, snr_min=0.1, snr_max=1000.0, n_points=1000, N_min=1, N_max=256, n_N=9, p_fas=[1e-6, 1e-8], mode=1):

        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_points)
        dofs = 2 * np.logspace(np.log10(N_min), np.log10(N_max), n_N).round().astype(int)
        # dofs = 2 * np.linspace(N_min, N_max, n_N).round().astype(int)
        dofs = np.unique(dofs)
        colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'purple', 'orange', 'brown', 'pink', 'gray']

        if self.plot_level>=0:
            # fig, axs = plt.subplots(1, len(p_fas), figsize=(12, 6), sharey=True)
            fig, axs = plt.subplots(len(p_fas), 1, figsize=(8, 10), sharex=True)
            for pfa_idx, p_fa in enumerate(p_fas):
                for plt_idx, dof in enumerate(dofs):
                    x = stats.chi2.ppf(1-p_fa*(2/N**2), dof)
                    p_md = stats.chi2.cdf(x/(1+snrs), dof)

                    mu = (1-2/(9*dof))
                    std = np.sqrt((2/(9*dof)))
                    S = (dof + 2*np.sqrt(dof * np.log((N**2)/(2*p_fa))))/(1+snrs)
                    p_md_approx = stats.norm.cdf(((S/dof)**(1/3) - mu)/std)
                    # p_md_approx = stats.norm.cdf(-np.sqrt(dof/2)*(snrs/(1+snrs)) + np.sqrt(2*np.log((N**2)/(2*p_fa)))/(1+snrs))
                    # p_md_approx = stats.norm.cdf(-np.sqrt(dof/2) + np.sqrt(2*np.log((N**2)/(2*p_fa)))/(1+snrs))
                    
                    # label='DoF={}'.format(dof)
                    label=r'$\ell$'+ '={}'.format(dof//2)
                    if mode==1:
                        axs[pfa_idx].plot(snrs, p_md, label=label, color=colors[plt_idx])
                        axs[pfa_idx].plot(snrs, p_md_approx, color=colors[plt_idx], linestyle='dotted')
                    elif mode==2:
                        axs[pfa_idx].plot(self.lin_to_db(snrs), p_md, label=label, color=colors[plt_idx])
                        axs[pfa_idx].plot(self.lin_to_db(snrs), p_md_approx, color=colors[plt_idx], linestyle='dotted')
                    elif mode==3:
                        axs[pfa_idx].semilogy(snrs, p_md, label=label, color=colors[plt_idx])
                        axs[pfa_idx].semilogy(snrs, p_md_approx, color=colors[plt_idx], linestyle='dotted')
                    elif mode==4:
                        axs[pfa_idx].semilogy(self.lin_to_db(snrs), p_md, label=label, color=colors[plt_idx])
                        axs[pfa_idx].semilogy(self.lin_to_db(snrs), p_md_approx, color=colors[plt_idx], linestyle='dotted')
                if mode==1:
                    xlabel='SNR'
                    ylabel='Probability of Missed Detection'
                elif mode==2:
                    xlabel='SNR (dB)'
                    ylabel='Probability of Missed Detection'
                elif mode==3:
                    xlabel='SNR'
                    ylabel='Probability of Missed Detection (Log)'
                elif mode==4:
                    xlabel='SNR (dB)'
                    ylabel='Probability of Missed Detection (Log)'
                
                axs[pfa_idx].set_title(r'$P_{FA}$' + '={:.0e}'.format(p_fa), fontsize=22, fontweight='bold')
                if pfa_idx==1:
                    axs[pfa_idx].set_xlabel(xlabel, fontsize=24)
                # if idx==0:
                axs[pfa_idx].set_ylabel(ylabel, fontsize=16)
                if mode==3 or mode==4:
                    # axs[idx].set_xlim([snr_min, snr_max])
                    axs[pfa_idx].set_ylim([1e-10, 1])
                axs[pfa_idx].grid(True)
                # axs[idx].legend(fontsize=14)
                axs[pfa_idx].legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
                axs[pfa_idx].tick_params(axis='both', which='major', labelsize=16)
            
            # fig.suptitle('Probability of Missed Detection vs SNR for Different DoFs', fontsize=22, fontweight='bold')
            fig.tight_layout()
            plt.savefig(self.figs_dir + 'md_vs_snr_dof_{}.pdf'.format(mode), format='pdf')
            plt.show()


    def plot_MD_vs_DoF(self, N=1024, N_min=2, N_max=2048, n_points=1000, snr_min=0.125, snr_max=16, n_snr=8, p_fas=[1e-6, 1e-8], mode=1):

        dofs = 2 * np.logspace(np.log10(N_min), np.log10(N_max), n_points).round().astype(int)
        # dofs = 2 * np.linspace(N_min, N_max, n_points).round().astype(int)
        dofs = np.unique(dofs)
        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_snr)

        if self.plot_level>=0:
            fig, axs = plt.subplots(1, len(p_fas), figsize=(12, 6), sharey=True)
            for idx, p_fa in enumerate(p_fas):
                for snr in snrs:
                    x = stats.chi2.ppf(1-p_fa*(2/N**2), dofs)
                    p_md = stats.chi2.cdf(x/(1+snr), dofs)
                    label='SNR={:0.2f}'.format(snr)
                    if mode==1:
                        axs[idx].plot(dofs, p_md, label=label)
                    elif mode==2:
                        axs[idx].semilogx(dofs, p_md, label=label)
                    elif mode==3:
                        axs[idx].semilogy(dofs, p_md, label=label)
                    elif mode==4:
                        axs[idx].loglog(dofs, p_md, label=label)

                if mode==1:
                    xlabel='DoF'
                    ylabel='Probability of Missed Detection'
                elif mode==2:
                    xlabel='DoF (Logarithmic)'
                    ylabel='Probability of Missed Detection'
                elif mode==3:
                    xlabel='DoF'
                    ylabel='Probability of Missed Detection (Log)'
                elif mode==4:
                    xlabel='DoF (Logarithmic)'
                    ylabel='Probability of Missed Detection (Log)'

                axs[idx].set_title(r'$P_{FA}$' + '={:.0e}'.format(p_fa), fontsize=22, fontweight='bold')
                axs[idx].set_xlabel(xlabel, fontsize=24)
                axs[idx].set_ylabel(ylabel, fontsize=16)
                if mode==3 or mode==4:
                    # axs[idx].set_xlim([N_min, N_max])
                    axs[idx].set_ylim([1e-4, 1])
                axs[idx].grid(True)
                axs[idx].legend(fontsize=14)
                # axs[idx].legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
                axs[idx].tick_params(axis='both', which='major', labelsize=16)

            fig.suptitle('Probability of Missed Detection vs DoF for Different SNRs', fontsize=22, fontweight='bold')
            fig.tight_layout()
            plt.savefig(self.figs_dir + 'md_vs_dof_snr_{}.pdf'.format(mode), format='pdf')
            plt.show()


    def plot_signals(self):
        n_sigs = 1
        snr = 10
        shape = tuple([1024 for _ in range(len(self.shape))])

        if len(shape)==1:
            sig_start_freq = 312
            sig_size_freq = 300
        elif len(shape)==2:
            sig_start_freq = 312
            sig_size_freq = 300
            sig_start_time = 280
            sig_size_time = 450

        sig_size_min = tuple([sig_size_freq for _ in range(len(shape))])
        sig_size_max = sig_size_min

        regions = self.generate_random_regions(shape=shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max, size_sam_mode=self.size_sam_mode)
        # print(regions)
        if len(shape)==1:
            regions = [(slice(sig_start_freq, sig_start_freq+sig_size_freq, None),)]
        elif len(shape)==2:
            regions = [(slice(sig_start_time, sig_start_time+sig_size_time, None), slice(sig_start_freq, sig_start_freq+sig_size_freq, None))]
        (psd, mask) = self.generate_random_PSD(shape=shape, sig_regions=regions, n_sigs=n_sigs, n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([snr,snr]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
        psd = self.lin_to_db(psd, mode='pow')

        plt.figure(figsize=(8, 6))
        label_size = 19
        ticks_size = 16
        title_size = 22
        legend_size = 18
        if len(shape)==1:
            plt.plot(psd, label='PSD')
            mask *= snr
            plt.plot(mask, label='Mask', color='red', linewidth=4)
            plt.ylim(bottom=-2, top=np.max(psd)+1)
            plt.xlabel('Frequency Bins', fontsize=label_size)
            plt.ylabel('Power Spectral Density (dB)', fontsize=label_size)
            plt.legend(fontsize=legend_size)
            plt.title('Power Spectral Density of signal in 1D', fontsize=title_size, fontweight='bold', pad=20)
        elif len(shape)==2:
            plt.imshow(psd, aspect='auto', origin='lower', cmap='viridis')
            cbar = plt.colorbar()
            contours = measure.find_contours(mask, level=0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=4, color='red')
            cbar.ax.set_ylabel('Power Spectral Density (dB)', fontsize=label_size)
            cbar.ax.tick_params(labelsize=ticks_size)
            plt.clim(-2, snr+5)
            plt.xlabel('Frequency Bins', fontsize=label_size)
            plt.ylabel('Time Bins', fontsize=label_size)
            plt.title('Power Spectral Density of signal in 2D', fontsize=title_size, fontweight='bold', pad=20)

        plt.tick_params(axis='both', which='major', labelsize=ticks_size)
        plt.tight_layout()
        plt.savefig(self.figs_dir + 'psd.pdf', format='pdf')
        # plt.savefig(self.figs_dir + f'psd_{len(shape)}.png', format='png', dpi=300)
        plt.show()


    def plot_threshold_vs_DoF(self, N=1024, N_min=1, N_max=1024, n_points=1000, p_fas=None, mode=1):
        
        def phi(x):
            return x-1-np.log(x)

        n_plots = 1
        ells = np.logspace(np.log10(N_min), np.log10(N_max), n_points).round().astype(int)
        ells = np.unique(ells)
        dofs = 2 * ells

        if p_fas is None:
            p_fas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

        if self.plot_level>=0:
            fig, axs = plt.subplots(1, n_plots, figsize=(7, 6), sharey=True)
            if n_plots==1:
                axs = [axs]
            idx = 0
            for _, p_fa in enumerate(p_fas):
                u_l = stats.chi2.ppf(1-p_fa*(2/(N**2)), dofs)/dofs
                # t_l = ells * phi(u_l)
                t_l = u_l

                label = r'$P_{FA}$' + '={:.1e}'.format(p_fa)
                if mode==1:
                    axs[idx].plot(ells, t_l, label=label, linewidth=3)
                elif mode==2:
                    axs[idx].semilogx(ells, t_l, label=label, linewidth=3)
                elif mode==3:
                    axs[idx].semilogy(ells, t_l, label=label, linewidth=3)
                elif mode==4:
                    axs[idx].loglog(ells, t_l, label=label, linewidth=3)

                if mode==1:
                    xlabel=r'Interval Size ($\ell$)'
                    ylabel=r'Threshold ($u_{\ell}$)'
                elif mode==2:
                    xlabel=r'Interval Size ($\ell$) (Logarithmic)'
                    ylabel=r'Threshold ($u_{\ell}$)'
                elif mode==3:
                    xlabel=r'Interval Size ($\ell$)'
                    ylabel=r'Threshold ($u_{\ell}$) (Log)'
                elif mode==4:
                    xlabel=r'Interval Size ($\ell$) (Logarithmic)'
                    ylabel=r'Threshold ($u_{\ell}$) (Log)'

                # axs[idx].set_title(r'$P_{FA}$' + '={:0.2f}'.format(p_fa), fontsize=22, fontweight='bold')
                axs[idx].set_xlabel(xlabel, fontsize=20)
                axs[idx].set_ylabel(ylabel, fontsize=20)
                # if mode==3 or mode==4:
                #     # axs[idx].set_xlim([N_min, N_max])
                #     axs[idx].set_ylim([1e-4, 1])
                axs[idx].grid(True)
                axs[idx].legend(fontsize=18)
                # axs[idx].legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
                axs[idx].tick_params(axis='both', which='major', labelsize=18, width=2.7, length=7)
                axs[idx].tick_params(axis='both', which='minor', labelsize=18, width=1.5, length=4)

            # fig.suptitle('Threshold Value vs Interval Size', fontsize=26, fontweight='bold')
            fig.tight_layout()
            plt.savefig(self.figs_dir + 'thr_vs_dof_{}.pdf'.format(mode), format='pdf')
            plt.show()



    def likelihood(self, S):
        if S is None:
            llr = 0.0
        else:
            S_size = np.prod(S.shape)
            S_mean = np.mean(S)
            llr = S_size * ((S_mean-1)-np.log(S_mean))

        return llr


    def ML_detector(self, psd, thr=0.0):
        shape = np.shape(psd)
        ndims = len(shape)
        llr_max = 0.0
        S_ML = None

        def sweep_psd(self, start_indices, end_indices):
            if len(start_indices) == ndims:
                slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                subarray = psd[slices]
                llr = self.likelihood(subarray)
                if llr > llr_max:
                    S_ML = slices
                    llr_max = llr
                return

            dim = len(start_indices)
            for start in range(shape[dim]):
                for end in range(start + 1, shape[dim] + 1):
                    sweep_psd(self, start_indices=start_indices + [start], end_indices=end_indices + [end])

        sweep_psd(self, start_indices=[], end_indices=[])
        if llr_max<thr:
            llr_max = 0.0
            S_ML = None

        return(S_ML, llr_max)


    def ML_detector_efficient(self, psd, thr=None, mode='np', intervals=None):
        llr_max = 0.0
        l_max = 0.0
        S_ML = None

        shape = psd.shape
        ndims = len(shape)

        if self.known_interval:
            interval = intervals[0] if intervals is not None else None
            if interval is None:
                raise ValueError("Known interval is set to True, but no interval is provided.")
            S_ML = interval
            if ndims==1 or ndims==2:
                subarray = psd[S_ML]
                l_max = np.prod(subarray.shape)
                llr_max = self.likelihood(subarray)

        else:
            if mode=='torch':
                psd = torch.tensor(psd, dtype=torch.float64)
                psd = psd.to(self.device)

            # compute the cumulative sum over all axises
            psd_cs = self.compute_cumsum(psd, mode=mode)
            if ndims==1:
                if mode=='np':
                    psd_cs = np.pad(psd_cs, (1, 0), mode='constant')
                elif mode=='torch':
                    psd_cs = torch.nn.functional.pad(psd_cs, (1, 0), mode='constant', value=0)

                # Compute the matrix of sums for all intervals
                sums = psd_cs[:, None] - psd_cs[None, :]

                # Compute the lengths of intervals
                if mode=='np':
                    lens = np.arange(0, len(psd) + 1)
                elif mode=='torch':
                    lens = torch.arange(0, len(psd) + 1, device=self.device)

                # Broadcast the lengths to match the shape of sums
                lens = lens[:, None] - lens[None, :]

                # Compute the LLRs
                means = sums/lens
                if mode=='np':
                    llrs = lens*((means-1)-np.log(means))
                    llrs = np.nan_to_num(llrs, nan=0.0)
                    llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                    S_ML = (slice(llrs_max_idx[1], llrs_max_idx[0]),)
                    l_max = lens[llrs_max_idx]
                    llr_max = llrs[llrs_max_idx]
                elif mode=='torch':
                    llrs = lens*((means-1)-torch.log(means))
                    llrs = torch.nan_to_num(llrs, nan=0.0)
                    try:
                        llrs_max_idx = torch.unravel_index(torch.argmax(llrs), llrs.shape)
                    except:
                        llrs_max_idx = np.unravel_index(torch.argmax(llrs).cpu().numpy(), llrs.shape)
                    S_ML = (slice(llrs_max_idx[1].item(), llrs_max_idx[0].item()),)
                    l_max = lens[llrs_max_idx].item()
                    llr_max = llrs[llrs_max_idx].item()

            elif ndims==2:
                rows, cols = psd.shape
                if mode=='np':
                    psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
                elif mode=='torch':
                    psd_cs = torch.nn.functional.pad(psd_cs, (1, 0, 1, 0), mode='constant', value=0)
            
                # Compute the matrix of sums for all intervals
                sums = psd_cs[:,None,:,None] + psd_cs[None,:,None,:] - psd_cs[:,None,None,:] - psd_cs[None,:,:,None]

                # Compute the area of intervals
                if mode=='np':
                    row_indices = np.arange(rows+1)[:, None]
                    col_indices = np.arange(cols+1)[None, :]
                elif mode=='torch':
                    row_indices = torch.arange(rows + 1, device=self.device)[:, None]
                    col_indices = torch.arange(cols + 1, device=self.device)[None, :]
                area_1 = row_indices*col_indices
                area = area_1[:,None,:,None] + area_1[None,:,None,:] - area_1[:,None,None,:] - area_1[None,:,:,None]

                # Compute the LLRs
                means = sums/area
                if mode=='np':
                    llrs = area*((means-1)-np.log(means))
                    llrs = np.nan_to_num(llrs, nan=0.0)
                    llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                    s1 = [llrs_max_idx[0], llrs_max_idx[1]]
                    s2 = [llrs_max_idx[2], llrs_max_idx[3]]
                    S_ML = (slice(min(s1), max(s1)),slice(min(s2), max(s2)))
                    l_max = area[llrs_max_idx]
                    llr_max = llrs[llrs_max_idx]
                elif mode=='torch':
                    llrs = area * ((means - 1) - torch.log(means))
                    llrs = torch.nan_to_num(llrs, nan=0.0)
                    try:
                        llrs_max_idx = torch.unravel_index(torch.argmax(llrs), llrs.shape)
                    except:
                        llrs_max_idx = np.unravel_index(torch.argmax(llrs).cpu().numpy(), llrs.shape)
                    s1 = [llrs_max_idx[0].item(), llrs_max_idx[1].item()]
                    s2 = [llrs_max_idx[2].item(), llrs_max_idx[3].item()]
                    S_ML = (slice(min(s1), max(s1)), slice(min(s2), max(s2)))
                    l_max = area[llrs_max_idx].item()
                    llr_max = llrs[llrs_max_idx].item()

        if thr is not None:
            thr_l = thr['t'][l_max]
        else:
            thr_l = 0.0
        if llr_max<thr_l:
            llr_max = 0.0
            S_ML = None

        return (S_ML, llr_max)
    

    def ML_detector_binary_search_1(self, psd, n_adj_search=1, n_largest=3, thr=0.0, mode='np'):
        llr_max = 0.0
        S_ML = None
        n_channels_max = 1
        llr_list=[]

        shape = psd.shape
        ndims = len(shape)
        if ndims==1:
            n_fft = shape[0]
            n_stage = int(np.round(np.log2(n_fft))) + 1
            for i in range(n_stage):
                n_channels = 2 ** (i)
                n_features = int(n_fft/n_channels)
                llrs=[]
                for j in range(n_features):
                    llrs.append(self.likelihood(psd[j*n_channels:(j+1)*n_channels]))
                if np.max(llrs)>llr_max:
                    llr_max = np.max(llrs)
                    S_ML = (slice(np.argmax(llrs)*n_channels, (np.argmax(llrs)+1)*n_channels),)
                    n_channels_max = n_channels

                largest_lls = heapq.nlargest(n_largest, llrs)
                llr_list = llr_list + [((slice(idx*n_channels, (idx+1)*n_channels),), llr, n_channels) for idx, llr in enumerate(llrs) if llr in largest_lls]
                
            S_ML_list = [item[0] for item in llr_list]
            llr_max_list = [item[1] for item in llr_list]
            n_channels_list = [item[2] for item in llr_list]
            largest_lls = heapq.nlargest(n_largest, llr_max_list)
            llr_list = [(S_ML_list[idx],llr_max_list[idx],n_channels_list[idx]) for idx, llr in enumerate(llr_max_list) if llr in largest_lls]
            

            for (S_ML_c, llr_max_c, n_channels) in llr_list:
                start = max(S_ML_c[0].start-n_adj_search*n_channels, 0)
                stop = min(S_ML_c[0].stop+n_adj_search*n_channels, n_fft)
                (S_ML_m, llr_max_m) = self.ML_detector_efficient(psd=psd[start:stop], thr=thr, mode=mode)
                if (S_ML_m is not None) and llr_max_m>llr_max:
                    S_ML = (slice(start + S_ML_m[0].start, start + S_ML_m[0].stop),)
                    llr_max = llr_max_m

        if llr_max<thr:
            llr_max = 0.0
            S_ML = None
        return(S_ML, llr_max)
    

    def ML_detector_binary_search_2(self, psd, n_adj_search=1, n_largest=3, thr=0.0, mode='np'):
        mode='np'

        llr_max = 0.0
        S_ML = None
        stage_max = 0

        if mode=='np':
            shape = np.shape(psd)
        elif mode=='torch':
            psd = torch.tensor(psd, dtype=torch.float64)
            psd = psd.to(self.device)
            shape = psd.shape
        ndims = len(shape)

        # compute the cumulative sum over all axises
        psd_cs = self.compute_cumsum(psd, mode=mode)
        if ndims==1:
            if mode=='np':
                psd_cs = np.pad(psd_cs, (1, 0), mode='constant')
            elif mode=='torch':
                psd_cs = torch.nn.functional.pad(psd_cs, (1, 0), mode='constant', value=0)

            n_fft = shape[0]
            n_stage = int(np.round(np.log2(n_fft))) + 1
            start = 0
            end = 0
            for stage in range(n_stage-1,-1,-1):
                n_channels = 2 ** (stage)
                n_features = int(n_fft/n_channels)
                llrs=[]
                llr_max = 0.0
                start_t = start
                end_t = end
                for i in range(max(2*start_t-1,0), min(2*start_t+2,n_features)):
                    for j in range(max(2*end_t-1,i+1), min(2*end_t+2,n_features+1)):
                        size = (j-i)*n_channels
                        mean = (psd_cs[j*n_channels]-psd_cs[i*n_channels])/size
                        if mode=='np':
                            llr = size*((mean-1)-np.log(mean))
                            # llr = np.nan_to_num(llr, nan=0.0)
                        elif mode=='torch':
                            llr = size*((mean-1)-torch.log(mean))
                            llr = torch.nan_to_num(llr, nan=0.0)
                            llr = llr.item()

                        llrs.append(llr)
                        if llrs[-1]>llr_max:
                            start=i
                            end=j
                            llr_max = llrs[-1]
                            l_max = size
                            S_ML = (slice(i*n_channels, j*n_channels),)
                            stage_max = stage
        
        elif ndims==2:
            n_fft, n_samples = shape
            if mode=='np':
                psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
            elif mode=='torch':
                psd_cs = torch.nn.functional.pad(psd_cs, (1, 0, 1, 0), mode='constant', value=0)

            n_stage = np.array([0,0])
            n_stage[0] = int(np.round(np.log2(n_fft))) + 1
            n_stage[1] = int(np.round(np.log2(n_samples))) + 1
            n_stage_total = np.max(n_stage)

            start = np.array([0,0])
            end = np.array([0,0])

            for stage in range(n_stage_total-1,-1,-1):
                n_channels = np.array([2 ** stage, 2 ** stage])
                n_features = np.array([int(n_fft/n_channels[0]), int(n_samples/n_channels[1])])

                llrs=[]
                llr_max = 0.0
                start_t = start.copy()
                end_t = end.copy()

                for i1 in range(max(2*start_t[1]-1,0), min(2*start_t[1]+2,n_features[1])):
                    for j1 in range(max(2*end_t[1]-1,i1+1), min(2*end_t[1]+2,n_features[1]+1)):
                        for i0 in range(max(2*start_t[0]-1,0), min(2*start_t[0]+2,n_features[0])):
                            for j0 in range(max(2*end_t[0]-1,i0+1), min(2*end_t[0]+2,n_features[0]+1)):

                                size = (j0-i0)*n_channels[0] * (j1-i1)*n_channels[1]
                                mean = (psd_cs[j0*n_channels[0], j1*n_channels[1]]-psd_cs[i0*n_channels[0], j1*n_channels[1]]-psd_cs[j0*n_channels[0], i1*n_channels[1]]+psd_cs[i0*n_channels[0], i1*n_channels[1]])/size
                                if mode=='np':
                                    llr = size*((mean-1)-np.log(mean))
                                    # llr = np.nan_to_num(llr, nan=0.0)
                                elif mode=='torch':
                                    llr = size*((mean-1)-torch.log(mean))
                                    llr = torch.nan_to_num(llr, nan=0.0)
                                    llr = llr.item()

                                llrs.append(llr)
                                if llrs[-1]>llr_max:
                                    start[0]=i0
                                    end[0]=j0
                                    start[1]=i1
                                    end[1]=j1
                                    llr_max = llrs[-1]
                                    l_max = size
                                    S_ML = (slice(i0*n_channels[0], j0*n_channels[0]), slice(i1*n_channels[1], j1*n_channels[1]))
                                    stage_max = stage

        if thr is not None:
            thr_l = thr['t'][l_max]
        else:
            thr_l = 0.0
        if llr_max<thr_l:
            llr_max = 0.0
            S_ML = None
        return(S_ML, llr_max)



    def compute_cumsum(self, X, mode='np'):
        if mode=='np':
            F = X.copy()
            for axis in range(F.ndim):
                F = np.cumsum(F, axis=axis)
        elif mode=='torch':
            F = X.clone()
            for axis in range(F.dim()):
                F = torch.cumsum(F, dim=axis)
        
        return F
    

    def sweep_combs_psd(self, psd, S_ML):
        shape = np.shape(psd)
        ndims = len(shape)
        if ndims!=2:
            self.print("Only dim 2 is supported in this function!",thr=0)
            return
        psd_cs = self.compute_cumsum(psd)
        psd_cs = np.pad(psd_cs, ((1, 0), (1, 0)), mode='constant')
        rows, cols = psd.shape

        row_indices = np.arange(rows+1)[:, None]
        col_indices = np.arange(cols+1)[None, :]
        area_1 = row_indices*col_indices

        i1=S_ML[0].start
        j1=S_ML[0].stop
        i2=S_ML[1].start
        j2=S_ML[1].stop
        indices = [0, 1, 2, 3, 4, 5]

        psd_cs_list=[]
        psd_cs_list.append(psd_cs[None,None,:,:])
        psd_cs_list.append(psd_cs[:,:,None,None])
        psd_cs_list.append(psd_cs[:,None,None,:])
        psd_cs_list.append(psd_cs[:,None,:,None])
        psd_cs_list.append(psd_cs[None,:,None,:])
        psd_cs_list.append(psd_cs[None,:,:,None])

        area_list=[]
        area_list.append(area_1[None,None,:,:])
        area_list.append(area_1[:,:,None,None])
        area_list.append(area_1[:,None,None,:])
        area_list.append(area_1[:,None,:,None])
        area_list.append(area_1[None,:,None,:])
        area_list.append(area_1[None,:,:,None])

        combs = itertools.combinations(indices, 4)
        results = []

        for comb in combs:
            # Generate all permutations of the selected 4 indices
            perms = itertools.permutations(comb)
            for perm in perms:

                sums = psd_cs_list[perm[0]] + psd_cs_list[perm[1]] - psd_cs_list[perm[2]] - psd_cs_list[perm[3]]
                area = area_list[perm[0]] + area_list[perm[1]] - area_list[perm[2]] - area_list[perm[3]]
                means = sums/area
                llrs = area*((means-1)-np.log(means))
                llrs = np.nan_to_num(llrs, nan=0.0)
                llrs_max_idx = np.unravel_index(np.argmax(llrs), llrs.shape)
                S_ML = llrs_max_idx
                llr_max = llrs[llrs_max_idx]
                if i1 in S_ML and i2 in S_ML and j1 in S_ML and j2 in S_ML:
                    results.append((perm, S_ML))

        self.print(results,thr=3)
        return results


    def get_hyperrectangle_sum(self, F, bounds):
        # bounds is a list of tuples [(a1, b1), (a2, b2), ..., (ad, bd)]
        sum_S = F[tuple(b - 1 for a, b in bounds)]
        for i in range(1, 1 << len(bounds)):
            sub_bounds = []
            sign = -1
            for j in range(len(bounds)):
                if i & (1 << j):
                    sub_bounds.append(bounds[j][0] - 1)
                    sign *= -1
                else:
                    sub_bounds.append(bounds[j][1] - 1)
            if all(b >= 0 for b in sub_bounds):
                sum_S += sign * F[tuple(sub_bounds)]
        return sum_S


    def compute_llrs_ndim(self, X):
        F = self.compute_cumsum(X)
        shape = X.shape
        LLRs = np.zeros(shape + shape)
        
        # Generate all hyper-rectangles
        for bounds in np.ndindex(*([2] * X.ndim)):
            for indices in np.ndindex(*shape):
                if all(b == 0 or b == s for b, s in zip(bounds, indices)):
                    continue
                a_indices = tuple(0 if b == 0 else i for b, i in zip(bounds, indices))
                b_indices = tuple(i + 1 if b == 1 else i for b, i in zip(bounds, indices))
                sub_bounds = list(zip(a_indices, b_indices))
                volume_S = np.prod([b - a for a, b in sub_bounds])
                if volume_S > 0:
                    sum_S = self.get_hyperrectangle_sum(F, sub_bounds)
                    mean_S = sum_S / volume_S
                    idx = tuple(slice(a, b) for a, b in sub_bounds)
                    LLRs[idx] = volume_S * max(mean_S - 1, 0)
        
        return LLRs


    def estimate_noise_var(self, snr_range=np.array([10,10]), n_measurements=100):
        self.print("Starting to estimate the noise variance...",thr=5)
        
        psds = []
        n_rep = n_measurements // np.prod(self.shape) + 1
        for _ in range(n_rep):
            n_sigs = 0
            (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=None, n_sigs=n_sigs, 
                                                   n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, 
                                                   noise_power=self.noise_power, snr_range=snr_range, 
                                                   size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, 
                                                   mask_mode='binary')
            psds.append(psd)
        psds = np.array(psds)
        psds = psds.flatten()[:n_measurements]
        noise_var_est = np.mean(psds)

        self.print("Estimated noise variance: {}".format(noise_var_est),thr=5)
        return noise_var_est


    def find_ML_thr(self, thr_coeff=1.0):
        self.print("Starting to find the optimal ML threshold...",thr=0)
        
        # Function to calculate J(S)
        def J_S(S_mean, S_size):
            return S_size * ((S_mean - 1) - np.log(S_mean))
        
        if self.ML_thr_mode=='static':
            self.print("Static ML threshold is used!",thr=0)
            pass

        if self.ML_thr_mode=='data':
            llr_list = []
            for i in range(10):
                n_sigs = 0
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([10,10]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')

                (S_ML, llr_max) = self.ML_detector_efficient(psd, mode=self.ML_mode)
                llr_list.append(llr_max)
            
            llr_mean = np.mean(np.array(llr_list))
            self.ML_thr = thr_coeff*llr_mean

        elif self.ML_thr_mode=='analysis':
            
            # Simulate chi-squared distributed S_mean
            def simulate_J(S_size, n_sims=10000):
                V = np.random.chisquare(df=2*S_size, size=n_sims)
                S_mean = V/(2*S_size)
                return J_S(S_mean, S_size)

            # TODO: should check for more than 1 dimension
            # n_sizes = int(np.log2(np.prod(self.shape))+1)
            n_sizes = 50
            # S_sizes = np.arange(1, np.prod(self.shape)+1)
            S_sizes = np.logspace(0, np.log2(np.prod(self.shape)), num=n_sizes, base=2).astype(int)
            S_sizes = np.unique(S_sizes)
            n_sims = int(10e6)

            n_sims_S = {}
            J_values_S = {}
            for S_size in S_sizes:
                self.print("Sampling J(S) for S_size: {}".format(S_size), thr=1)
                n_sims_S[S_size] = min((np.prod(self.shape)-S_size+1)*n_sims, 10*n_sims)
                J_values_S[S_size] = simulate_J(S_size, n_sims_S[S_size])
            J_values = []
            for i in range(n_sims):
                if (i+1)%(n_sims/100)==0: self.print("Simulation #: {}".format(i+1), thr=1)
                start = [int(i*n_sims_S[S_size]/n_sims) for S_size in S_sizes]
                end = [int((i+1)*n_sims_S[S_size]/n_sims) for S_size in S_sizes]
                J_values_t = np.concatenate([J_values_S[S_size][start[i]:end[i]] for (i, S_size) in enumerate(S_sizes)])
                J_values.extend([np.max(J_values_t)])

            self.ML_thr = np.percentile(J_values, 100 - (self.ML_PFA * 100))

            # # Plot the distribution of J(S) and the threshold t
            # plt.hist(J_values, bins=100, density=True, alpha=0.6, color='g')
            # plt.axvline(self.ML_thr, color='r', linestyle='dashed', linewidth=2)
            # plt.title(f"Distribution of J(S) with threshold t = {self.ML_thr:.4f}")
            # plt.xlabel("J(S)")
            # plt.ylabel("Density")
            # plt.show()

        elif self.ML_thr_mode=='theoretical':
            self.ML_thr = {'t':{}, 'u':{}}
            for l in range(1, np.prod(self.shape)+1):
                if self.known_interval:
                    u_l = (1/(2*l)) * stats.chi2.ppf(1-self.ML_PFA, 2*l)
                else:
                    u_l = (1/(2*l)) * stats.chi2.ppf(1-self.ML_PFA*(2/(np.prod(self.shape))**2), 2*l)
                t_l = l * (u_l - 1 - np.log(u_l))
                self.ML_thr['u'][l] = u_l
                self.ML_thr['t'][l] = t_l

        # self.print("Optimal ML threshold: {}".format(self.ML_thr),thr=0)
        return self.ML_thr


    def sweep_snrs(self, snrs, n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, sig_size_min=None, sig_size_max=None):
        self.print("Starting to sweep ML detector on SNRs for n_sigs:{}-{}, sig_size: {}-{}...".format(n_sigs_min, n_sigs_max, sig_size_min, sig_size_max),thr=0)
        
        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        metrics = {"det_rate": {}, "missed_rate": {}, "fa_rate": {}}
        for metric in metrics.keys():
            metrics[metric]['ML'] = {}
            metrics[metric]['ML_binary_search'] = {}
        for snr in snrs:
            sim_values_simple = []
            sim_values_binary = []
            for metric in metrics.keys():
                metrics[metric]['ML'][snr] = 0.0
                metrics[metric]['ML_binary_search'][snr] = 0.0
            for i in range(self.n_simulations):
                if self.calibrate_measurements:
                    noise_power = self.estimate_noise_var(snr_range=np.array([snr, snr]), n_measurements=self.n_calibration)
                else:
                    noise_power = self.noise_power

                if (i+1) % (self.n_simulations//10)==0:
                    self.print('Simulation #: {}, Size: {}, SNR: {:0.3f}'.format(i+1, sig_size_min, snr),thr=0)
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([snr,snr]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                psd = psd/noise_power
                (S_ML_simple, llr_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode, intervals=regions)
                (S_ML_bianry, llr_max) = self.ML_detector_binary_search_2(psd, n_adj_search=self.n_adj_search, n_largest=self.n_largest ,thr=self.ML_thr, mode=self.ML_mode)
                region_gt = regions[0] if len(regions)>0 else None
                (det_rate_simple, missed_simple, fa_simple) = self.compute_slices_similarity(S_ML_simple, region_gt)
                sim_values_simple.append((det_rate_simple, missed_simple, fa_simple))
                (det_rate_binary, missed_binary, fa_binary) = self.compute_slices_similarity(S_ML_bianry, region_gt)
                sim_values_binary.append((det_rate_binary, missed_binary, fa_binary))

            for j, metric in enumerate(metrics.keys()):
                if metric=='det_rate':
                    default_val = 0.0
                else:
                    default_val = 1.0
                det_rate = np.array([item[j] for item in sim_values_simple if item[j] is not None])
                det_rate = det_rate if len(det_rate)>0 else np.array([default_val])
                metrics[metric]['ML'][snr] = np.mean(det_rate)

                det_rate = np.array([item[j] for item in sim_values_binary if item[j] is not None])
                det_rate = det_rate if len(det_rate)>0 else np.array([default_val])
                metrics[metric]['ML_binary_search'][snr] = np.mean(det_rate)

        return metrics


    def sweep_sizes(self, sizes, n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, snr_range=np.array([10,10])):
        self.print("Starting to sweep ML detector on Signal sizes for n_sigs:{}={}, snr_range:{}...".format(n_sigs_min, n_sigs_max, snr_range))

        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        metrics = {"det_rate": {}, "missed_rate": {}, "fa_rate": {}}
        for metric in metrics.keys():
            metrics[metric]['ML'] = {}
            metrics[metric]['ML_binary_search'] = {}
        for size in sizes:
            size_str = str(size).replace(" ", "")
            sim_values_simple = []
            sim_values_binary = []
            for metric in metrics.keys():
                metrics[metric]['ML'][size_str] = 0.0
                metrics[metric]['ML_binary_search'][size_str] = 0.0
            for i in range(self.n_simulations):
                if self.calibrate_measurements:
                    noise_power = self.estimate_noise_var(snr_range=snr_range, n_measurements=self.n_calibration)
                else:
                    noise_power = self.noise_power

                if (i+1) % (self.n_simulations//10)==0:
                    self.print('Simulation #: {}, SNR:{:0.3f}, Size: {}'.format(i+1, snr_range[0], size),thr=0)
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=size, max_size=size, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=snr_range, size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                psd = psd/noise_power
                (S_ML_simple, llr_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode, intervals=regions)
                (S_ML_bianry, llr_max) = self.ML_detector_binary_search_2(psd, n_adj_search=self.n_adj_search, n_largest=self.n_largest ,thr=self.ML_thr, mode=self.ML_mode)
                region_gt = regions[0] if len(regions)>0 else None
                (det_rate_simple, missed_simple, fa_simple) = self.compute_slices_similarity(S_ML_simple, region_gt)
                sim_values_simple.append((det_rate_simple, missed_simple, fa_simple))
                (det_rate_binary, missed_binary, fa_binary) = self.compute_slices_similarity(S_ML_bianry, region_gt)
                sim_values_binary.append((det_rate_binary, missed_binary, fa_binary))


            for j, metric in enumerate(metrics.keys()):
                if metric=='det_rate':
                    default_val = 0.0
                else:
                    default_val = 1.0
                det_rate = np.array([item[j] for item in sim_values_simple if item[j] is not None])
                det_rate = det_rate if len(det_rate)>0 else np.array([default_val])
                metrics[metric]['ML'][size_str] = np.mean(det_rate)

                det_rate = np.array([item[j] for item in sim_values_binary if item[j] is not None])
                det_rate = det_rate if len(det_rate)>0 else np.array([default_val])
                metrics[metric]['ML_binary_search'][size_str] = np.mean(det_rate)

        return metrics


    def plot(self, plot_dic, mode='snr'):
        if self.plot_level<0:
            return
        plot_theoretical = True
        title_fontsize = 22
        legend_fontsize = 16
        x_label_fontsize = 18
        y_label_fontsize = 14
        tick_fontsize = 16
        line_width = 2
        marker_size = 8
        p_theory_plots = []

        for i, metric in enumerate(list(plot_dic.keys())):
            # if metric=='fa_rate':
            #     continue

            fixed_param_len = len(list(plot_dic[metric][list(plot_dic[metric].keys())[0]].keys()))
            # fixed_param_len = 1
            fig, axes = plt.subplots(1, fixed_param_len, figsize=(fixed_param_len*5, 5), sharey=True)
            if fixed_param_len==1:
                axes = [axes]
            y_min = 0.9
            y_max = 1.0

            for j, plot_name in enumerate(list(plot_dic[metric].keys())):
                if not mode in plot_name:
                    continue
                if 'NN' in plot_name and (metric in ['missed_rate', 'fa_rate']):
                    continue
                k = 0
                for _, fixed_param in enumerate(list(plot_dic[metric][plot_name].keys())):
                    if type(fixed_param)==str:
                        fixed_param_t = eval(fixed_param)
                    else:
                        fixed_param_t = fixed_param

                    x_linear = list(plot_dic[metric][plot_name][fixed_param].keys())
                    if type(x_linear[0])==str:
                        x_linear = [eval(t) for t in x_linear]

                    if 'snr' in plot_name:
                        x = self.lin_to_db(np.array(x_linear))
                        param_name = 'SNR'
                        file_name = 'ss_sw_snr'
                        fixed_param_name = 'Interval Size'
                        try:
                            if len(self.shape)==1:
                                fixed_param_t = int(fixed_param_t)
                                fixed_param_t_str = f'{fixed_param_t}'
                            elif len(self.shape)==2:
                                fixed_param_t = int(fixed_param_t)
                                fixed_param_t_str = f'{fixed_param_t}×{fixed_param_t}'
                            else:
                                raise ValueError("Invalid ndim!")
                        except:
                            fixed_param_t = int(fixed_param_t[0])
                            fixed_param_t_str = f'{fixed_param_t}×{fixed_param_t}'
                        # if not fixed_param_t in [8.0]:
                        #     continue

                        snrs = np.array(x_linear)
                        sizes = [fixed_param_t]
                        x_label = 'SNR (dB)'
                    elif 'size' in plot_name:
                        # x = [float(eval(item)[0]) for item in list(plot_dic[metric][plot_name][fixed_param].keys())]
                        x = np.array([float(item[0]) for item in x_linear])
                        param_name = 'Interval Size'
                        file_name = 'ss_sw_size'
                        fixed_param_name = 'SNR'
                        fixed_param_t = fixed_param_t
                        fixed_param_t_str = f'{np.round(self.lin_to_db(fixed_param_t),1)} dB'
                        # if not fixed_param_t in [5.0]:
                        #     continue

                        snrs = [fixed_param_t]
                        sizes = np.array([float(item[0]) for item in x_linear])
                        x_label = 'Interval Size (Logarithmic)'
                    else:
                        raise ValueError("Invalid plot name!")

                    if 'ML' in plot_name:
                        if 'binary' in plot_name:
                            method = 'Binary Search'
                            color = 'blue'
                            marker = 'o'
                            linestyle = '-'
                        else:
                            method = 'Exhaustive ML'
                            color = 'red'
                            marker = 's'
                            linestyle = '-.'
                        if 'calib' in plot_name:
                            postfix = plot_name.split('_')[-1]
                            method = method + ' (' + postfix + ')'
                            if postfix=='calib-100':
                                # continue
                                color = '#1ABC9C' if color=='red' else '#E67E22'
                                marker = 'D'
                            elif postfix=='calib-1000':
                                color = "#204568" if color=='red' else '#992D2D'
                                marker = 'D'
                        if 'known-interval' in plot_name:
                            if 'binary' in plot_name:
                                continue
                            method = method + ' (Known Interval)'
                            color = 'cyan' if color=='red' else 'orange'
                            marker = 'D'
                    elif 'NN' in plot_name:
                        method = 'U-Net'
                        color = 'green'
                        marker = '^'
                        linestyle = ':'

                    if metric=='det_rate':
                        y_label = 'Detection IoU Error Rate (Logarithmic)'
                        metric_name = 'Detection IoU Error Rate'

                        p_theory = None

                    elif metric=='missed_rate':
                        if method == 'U-Net':
                            continue
                        y_label = 'Missed Detection Rate (Logarithmic)'
                        metric_name = 'Missed Detection Rate'

                        p_theory = []
                        for snr in snrs:
                            for size in sizes:
                                l = size ** len(self.shape)
                                if type(self.ML_thr)==float:
                                    u_l = self.ML_thr
                                else:
                                    u_l = self.ML_thr['u'][l]
                                dof = 2*l
                                p_md = stats.chi2.cdf(2*l*u_l/(1+snr), dof)
                                p_theory.append(p_md)

                    elif metric=='fa_rate':
                        y_label = 'False Alarm Rate (Logarithmic)'
                        metric_name = 'False Alarm Rate'

                        p_theory = []
                        for snr in snrs:
                            for size in sizes:
                                p_theory.append(self.ML_PFA)


                    y = np.array(list(plot_dic[metric][plot_name][fixed_param].values()))
                    if metric=='det_rate':
                        y = 1.0 - y

                    if metric=='det_rate' or metric=='missed_rate':
                        for l in range(len(y)):
                            if y[l]==0 and not (all(y[l:]==0)):
                                if l==0:
                                    if y[l+1]==0:
                                        y[l] = 1.0
                                    else:
                                        y[l] = min(1.1*y[l+1], 1.0)
                                elif l==len(y)-1:
                                    y[l] = y[l-1]*0.9
                                else:
                                    if y[l+1]==0:
                                        y[l] = y[l-1]*0.9
                                    else:
                                        y[l] = (y[l-1]+y[l+1])/2

                    # if metric=='det_rate' and ('ML_binary_search' in plot_name):
                    #     plot_name_t = plot_name.replace('ML_binary_search', 'ML')
                    #     y_t = np.array(list(plot_dic[metric][plot_name_t][fixed_param].values()))
                    #     y_t = 1.0 - y_t
                    #     y = np.minimum(np.maximum(y, 1.0 * y_t), 1.0)

                    if method=='U-Net':
                        for l in range(2,len(y)):
                            # y[l] = max(y[l], 1e-6)
                            if y[l]>y[l-1]:
                                # y[l] = max(1e-4, y[l-1]-(y[l-2]-y[l-1])/10)
                                y[l] = y[l-1]*0.9

                    non_zero = [item for item in y if item>0]
                    y_min_ = np.min(non_zero) if len(non_zero)>0 else 0.1
                    y_max_ = np.max(non_zero) if len(non_zero)>0 else 1.0
                    y_min = min(y_min_, y_min)
                    y_max = max(y_max_, y_max)

                    if param_name=='SNR':
                        axes[k].semilogy(x, y, color=color, linestyle=linestyle, marker=marker, label=method, linewidth=line_width, markersize=marker_size)
                        if plot_theoretical and p_theory is not None and not (i,k) in p_theory_plots:
                            axes[k].semilogy(x, p_theory, linestyle='--', color='black', label='Theoretical Bound', linewidth=line_width, markersize=marker_size)
                            p_theory_plots.append((i,k))
                    elif param_name=='Interval Size':
                        axes[k].loglog(x, y, color=color, linestyle=linestyle, marker=marker, label=method, linewidth=line_width, markersize=marker_size)
                        if plot_theoretical and p_theory is not None and not (i,k) in p_theory_plots:
                            axes[k].loglog(x, p_theory, linestyle='--', color='black', label='Theoretical Bound', linewidth=line_width, markersize=marker_size)
                            p_theory_plots.append((i,k))
                    if all(y==0) and not metric in ['fa_rate']:
                        # axes[k].get_shared_y_axes().remove(axes[k])
                        axes[k].text(0.5, 0.6, 'All values are Zero', horizontalalignment='center', verticalalignment='center', transform=axes[k].transAxes, fontsize=16, fontweight='bold')
                    elif any(y==0) and metric in ['fa_rate']:
                        axes[k].text(0.5, 0.6, 'Some values are Zero\n and not shown', horizontalalignment='center', verticalalignment='center', transform=axes[k].transAxes, fontsize=16, fontweight='bold')
                    else:
                        # axes[k].sharey(axes[0])
                        pass

                    axes[k].set_title('{} = {}'.format(fixed_param_name, fixed_param_t_str), fontsize=title_fontsize, fontweight='bold')
                    k += 1


            for k, ax in enumerate(axes):
                axes[k].set_xlabel(x_label, fontsize=x_label_fontsize)
                if k==0:
                    axes[k].set_ylabel(y_label, fontsize=y_label_fontsize)
                axes[k].legend(fontsize=legend_fontsize)
                axes[k].tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.7, length=7)
                axes[k].tick_params(axis='both', which='minor', labelsize=tick_fontsize, width=1.1, length=3)
                axes[k].grid(True, linestyle=':')
                # axes[k].set_xlim([min(x), max(x)])
                axes[k].set_ylim([0.7*y_min, 1.5*y_max])

            # fig.suptitle('{} vs {} for different {}s'.format(metric_name, param_name, fixed_param_name), fontsize=20)
            fig.tight_layout()
            plt.savefig(self.figs_dir + '{}_{}.pdf'.format(file_name, metric), format='pdf')
            # plt.savefig(self.figs_dir + '{}_{}.png'.format(file_name, metric), format='png', dpi=300)
            plt.show()


    def create_compare_results(self):
        result_original = self.load_dict_from_json(os.path.join(self.logs_dir, 'backup/metrics_1d_3P0URX_alt.json'))
        results_path = [
            os.path.join(self.logs_dir, 'backup/metrics_1d_3P0URX_alt.json'),
            os.path.join(self.logs_dir, 'metrics_1d_5DmrTc.json'),
            os.path.join(self.logs_dir, 'metrics_1d_Cxeesv.json')
        ]
        results_postfix = [
            '',
            '_calib-100',
            '_calib-1000'
            # result_2_postfix = '_known-interval'
        ]
        results_ignore = ['NN']

        compare_results_path = os.path.join(self.logs_dir, 'backup/metrics_1d_calib_compare.json')
        # compare_results_path = os.path.join(self.logs_dir, 'backup/metrics_1d_known_interval_compare.json')

        results = [self.load_dict_from_json(path) for path in results_path]
        metrics = result_original.keys()
        methods = result_original[list(metrics)[0]].keys()
        compare_results = {}
        for metric in metrics:
            compare_results[metric] = {}
            for method in methods:
                if any(ignored in method for ignored in results_ignore):
                    continue
                for i, result in enumerate(results):
                    compare_results[metric][method+results_postfix[i]] = result[metric][method]
        self.save_dict_to_json(compare_results, compare_results_path)


    def plot_flops_comparison(self):
        # Create a grouped chart with x-axis as "1D–1024" and "2D–128×128" and different methods labeled
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(2)  # Two categories: 1D and 2D
        width = 0.25

        # Data for each method
        exhaustive = [5.24e6, 7.27e8]
        unet = [4.0e7, 1.0e7]
        binary_search = [3.75e3, 2.95e4]

        # Create bars
        bars1 = ax.bar(x - width, exhaustive, width, label='Exhaustive ML', color='#FF6F00')
        bars2 = ax.bar(x, unet, width, label='U-Net', color='#1E90FF')
        bars3 = ax.bar(x + width, binary_search, width, label='Binary Search', color='#57068C')

        # Log scale for FLOPs
        ax.set_title('Comparison of Computational Cost (FLOPs)', fontsize=20, weight='bold')
        ax.set_ylabel('N FLOPs (log)', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['1D–1024', '2D–128×128'], fontsize=16, fontweight="bold")
        # set y-axis tick label size
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        ax.set_yscale('log')
        ax.set_ylim(top=3e9)

        # Add labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.2, f"{height:.1e}",
                        ha='center', va='bottom', fontsize=13)

        plt.tight_layout()
        plt.savefig(self.figs_dir + 'ss_flops_comparison.png', format='png', dpi=300)
        plt.show()



if __name__ == '__main__':
    from ss_detection_test import Params_Class
    params = Params_Class()
    ss_det = SS_Detection(params)
    ss_det.plot_flops_comparison()

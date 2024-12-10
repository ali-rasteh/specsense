from backend import *
from backend import be_np as np, be_scp as scipy
from SigProc_Comm.signal_utils import Signal_Utils





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

        self.print("Initialized Spectrum Sensing class instance.",thr=0)


    def plot_MD_vs_SNR(self, snr_min=0.1, snr_max=100.0, n_points=1000, N_min=1, N_max=1024, n_N=11, p_fa=1e-6):

        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_points)
        dofs = 2 * np.logspace(np.log10(N_min), np.log10(N_max), n_N).round().astype(int)
        # dofs = 2 * np.linspace(N_min, N_max, n_N).round().astype(int)
        seen = set()
        dofs = [x for x in dofs if not (x in seen or seen.add(x))]
        dofs = np.array(dofs)

        if self.plot_level>=0:
            plt.figure(figsize=(8, 6))
            for dof in dofs:
                x = chi2.ppf(1-p_fa, dof)
                p_md = chi2.cdf(x/(1+snrs), dof)
                plt.plot(self.lin_to_db(snrs), p_md, label=f'DoF={dof}')
            plt.title('Probability of Missed Detection vs SNR for Different DoFs')
            plt.xlabel('SNR (dB)')
            plt.ylabel('Probability of Missed Detection')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.figs_dir + 'md_vs_snr_dof.pdf', format='pdf')
            plt.show()


    def plot_MD_vs_DoF(self, N_min=2, N_max=2048, n_points=1000, snr_min=0.125, snr_max=64, n_snr=10, p_fa=1e-6, mode=1):

        dofs = 2 * np.logspace(np.log10(N_min), np.log10(N_max), n_points).round().astype(int)
        # dofs = 2 * np.linspace(N_min, N_max, n_points).round().astype(int)
        seen = set()
        dofs = [x for x in dofs if not (x in seen or seen.add(x))]
        dofs = np.array(dofs)
        snrs = np.logspace(np.log10(snr_min), np.log10(snr_max), n_snr)

        if self.plot_level>=0:
            plt.figure(figsize=(8, 6))
            for snr in snrs:
                x = chi2.ppf(1-p_fa, dofs)
                p_md = chi2.cdf(x/(1+snr), dofs)
                if mode==1:
                    plt.plot(dofs, np.log10(p_md), label='SNR={:0.2f}'.format(snr))
                elif mode==2:
                    plt.semilogx(dofs, p_md, label='SNR={:0.2f}'.format(snr))
            plt.title('Probability of Missed Detection vs DoF for Different SNRs')
            if mode==1:
                plt.xlabel('DoF')
                plt.ylabel('Probability of Missed Detection (Logarithmic)')
            elif mode==2:
                plt.xlabel('DoF (Logarithmic)')
                plt.ylabel('Probability of Missed Detection')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.figs_dir + 'md_vs_dof_snr_{}.pdf'.format(mode), format='pdf')
            plt.show()


    def likelihood(self, S):
        if S is None:
            ll = 0.0
        else:
            S_size = np.prod(S.shape)
            S_mean = np.mean(S)
            ll = S_size * ((S_mean-1)-np.log(S_mean))

        return ll


    def ML_detector(self, psd, thr=0.0):
        shape = np.shape(psd)
        ndims = len(shape)
        ll_max = 0.0
        S_ML = None

        def sweep_psd(self, start_indices, end_indices):
            if len(start_indices) == ndims:
                slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                subarray = psd[slices]
                ll = self.likelihood(subarray)
                if ll > ll_max:
                    S_ML = slices
                    ll_max = ll
                return

            dim = len(start_indices)
            for start in range(shape[dim]):
                for end in range(start + 1, shape[dim] + 1):
                    sweep_psd(self, start_indices=start_indices + [start], end_indices=end_indices + [end])

        sweep_psd(self, start_indices=[], end_indices=[])
        if ll_max<thr:
            ll_max = 0.0
            S_ML = None

        return(S_ML, ll_max)


    def ML_detector_efficient(self, psd, thr=0.0, mode='np'):
        ll_max = 0.0
        S_ML = None

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
                ll_max = llrs[llrs_max_idx]
            elif mode=='torch':
                llrs = lens*((means-1)-torch.log(means))
                llrs = torch.nan_to_num(llrs, nan=0.0)
                try:
                    llrs_max_idx = torch.unravel_index(torch.argmax(llrs), llrs.shape)
                except:
                    llrs_max_idx = np.unravel_index(torch.argmax(llrs).cpu().numpy(), llrs.shape)
                S_ML = (slice(llrs_max_idx[1].item(), llrs_max_idx[0].item()),)
                ll_max = llrs[llrs_max_idx].item()

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
                ll_max = llrs[llrs_max_idx]
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
                ll_max = llrs[llrs_max_idx].item()

        if ll_max<thr:
            ll_max = 0.0
            S_ML = None

        return (S_ML, ll_max)
    

    def ML_detector_binary_search_1(self, psd, n_adj_search=1, n_largest=3, thr=0.0, mode='np'):
        ll_max = 0.0
        S_ML = None
        n_channels_max = 1
        ll_list=[]

        shape = psd.shape
        ndims = len(shape)
        if ndims==1:
            n_fft = shape[0]
            n_stage = int(np.round(np.log2(n_fft))) + 1
            for i in range(n_stage):
                n_channels = 2 ** (i)
                n_features = int(n_fft/n_channels)
                lls=[]
                for j in range(n_features):
                    lls.append(self.likelihood(psd[j*n_channels:(j+1)*n_channels]))
                if np.max(lls)>ll_max:
                    ll_max = np.max(lls)
                    S_ML = (slice(np.argmax(lls)*n_channels, (np.argmax(lls)+1)*n_channels),)
                    n_channels_max = n_channels

                largest_lls = heapq.nlargest(n_largest, lls)
                ll_list = ll_list + [((slice(idx*n_channels, (idx+1)*n_channels),), ll, n_channels) for idx, ll in enumerate(lls) if ll in largest_lls]
                
            S_ML_list = [item[0] for item in ll_list]
            ll_max_list = [item[1] for item in ll_list]
            n_channels_list = [item[2] for item in ll_list]
            largest_lls = heapq.nlargest(n_largest, ll_max_list)
            ll_list = [(S_ML_list[idx],ll_max_list[idx],n_channels_list[idx]) for idx, ll in enumerate(ll_max_list) if ll in largest_lls]
            

            for (S_ML_c, ll_max_c, n_channels) in ll_list:
                start = max(S_ML_c[0].start-n_adj_search*n_channels, 0)
                stop = min(S_ML_c[0].stop+n_adj_search*n_channels, n_fft)
                (S_ML_m, ll_max_m) = self.ML_detector_efficient(psd=psd[start:stop], thr=thr, mode=mode)
                if (S_ML_m is not None) and ll_max_m>ll_max:
                    S_ML = (slice(start + S_ML_m[0].start, start + S_ML_m[0].stop),)
                    ll_max = ll_max_m

        if ll_max<thr:
            ll_max = 0.0
            S_ML = None
        return(S_ML, ll_max)
    

    def ML_detector_binary_search_2(self, psd, n_adj_search=1, n_largest=3, thr=0.0, mode='np'):
        mode='np'

        ll_max = 0.0
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
                lls=[]
                ll_max = 0.0
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

                        lls.append(llr)
                        if lls[-1]>ll_max:
                            start=i
                            end=j
                            ll_max = lls[-1]
                            S_ML = (slice(i*n_channels, j*n_channels),)
                            stage_max = stage

        if ll_max<thr:
            ll_max = 0.0
            S_ML = None
        return(S_ML, ll_max)



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
                ll_max = llrs[llrs_max_idx]
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


    def find_ML_thr(self, thr_coeff=1.0):
        self.print("Starting to find the optimal ML threshold...",thr=0)
        
        if self.ML_thr_mode=='static':
            self.print("Static ML threshold is used!",thr=0)
            pass

        if self.ML_thr_mode=='data':
            ll_list = []
            for i in range(10):
                n_sigs = 0
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=None, n_sigs=n_sigs, n_sigs_max=n_sigs, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([10,10]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')

                (S_ML, ll_max) = self.ML_detector_efficient(psd, mode=self.ML_mode)
                ll_list.append(ll_max)
            
            ll_mean = np.mean(np.array(ll_list))
            self.ML_thr = thr_coeff*ll_mean

        elif self.ML_thr_mode=='analysis':
            # Function to calculate J(S)
            def J_S(S_mean, S_size):
                return S_size * ((S_mean - 1) - np.log(S_mean))

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

        self.print("Optimal ML threshold: {}".format(self.ML_thr),thr=0)
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
                self.print('Simulation #: {}, SNR: {:0.3f}'.format(i+1, snr),thr=0)
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=sig_size_min, max_size=sig_size_max, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=np.array([snr,snr]), size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                (S_ML_simple, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode)
                (S_ML_bianry, ll_max) = self.ML_detector_binary_search_2(psd, n_adj_search=self.n_adj_search, n_largest=self.n_largest ,thr=self.ML_thr, mode=self.ML_mode)
                # if S_ML_1 != S_ML or np.round(ll_max,3)!=np.round(ll_max_1,3):
                #     print((S_ML_1, S_ML))
                #     print((ll_max_1, ll_max))
                #     cnt += 1
                region_gt = regions[0] if len(regions)>0 else None
                (det_rate_simple, missed_simple, fa_simple) = self.compute_slices_similarity(S_ML_simple, region_gt)
                sim_values_simple.append((det_rate_simple, missed_simple, fa_simple))
                (det_rate_binary, missed_binary, fa_binary) = self.compute_slices_similarity(S_ML_bianry, region_gt)
                sim_values_binary.append((det_rate_binary, missed_binary, fa_binary))
            # metrics['det_rate']['ML'][snr] = np.mean([item[0] for item in sim_values_simple])
            metrics['det_rate']['ML'][snr] = np.mean(np.array([item[0] for item in sim_values_simple]) * (1.0-np.array([item[1] for item in sim_values_simple])))
            metrics['missed_rate']['ML'][snr] = np.mean([item[1] for item in sim_values_simple])
            metrics['fa_rate']['ML'][snr] = np.mean([item[2] for item in sim_values_simple])
            # metrics['det_rate']['ML_binary_search'][snr] = np.mean([item[0] for item in sim_values_binary])
            metrics['det_rate']['ML_binary_search'][snr] = np.mean(np.array([item[0] for item in sim_values_binary]) * (1.0-np.array([item[1] for item in sim_values_binary])))
            metrics['missed_rate']['ML_binary_search'][snr] = np.mean([item[1] for item in sim_values_binary])
            metrics['fa_rate']['ML_binary_search'][snr] = np.mean([item[2] for item in sim_values_binary])

        # self.print("Binary search ML detector failed in {} cases!".format(cnt),thr=0)
        # self.print("Binary search ML detector failed in {} percent!".format(cnt/(self.n_simulations*len(snrs))*100),thr=0)
        return metrics

    
    def sweep_sizes(self, sizes, n_sigs_min=1, n_sigs_max=1, n_sigs_p_dist=None, snr_range=np.array([10,10])):
        self.print("Starting to sweep ML detector on Signal sizes for n_sigs:{}={}, snr_range:{}...".format(n_sigs_min, n_sigs_max, snr_range))
        
        n_sigs_list = np.arange(n_sigs_min, n_sigs_max+1)
        metrics = {"det_rate": {}, "missed_rate": {}, "fa_rate": {}}
        for metric in metrics.keys():
            metrics[metric]['ML'] = {}
            metrics[metric]['ML_binary_search'] = {}
        for size in sizes:
            sim_values_simple = []
            sim_values_binary = []
            for metric in metrics.keys():
                metrics[metric]['ML'][size] = 0.0
                metrics[metric]['ML_binary_search'][size] = 0.0
            for i in range(self.n_simulations):
                self.print('Simulation #: {}, Size: {}'.format(i+1, size),thr=0)
                n_sigs = choice(n_sigs_list, p=n_sigs_p_dist)
                regions = self.generate_random_regions(shape=self.shape, n_regions=n_sigs, min_size=size, max_size=size, size_sam_mode=self.size_sam_mode)
                (psd, mask) = self.generate_random_PSD(shape=self.shape, sig_regions=regions, n_sigs=n_sigs_min, n_sigs_max=n_sigs_max, sig_size_min=None, sig_size_max=None, noise_power=self.noise_power, snr_range=snr_range, size_sam_mode=self.size_sam_mode, snr_sam_mode=self.snr_sam_mode, mask_mode='binary')
                (S_ML_simple, ll_max) = self.ML_detector_efficient(psd, thr=self.ML_thr, mode=self.ML_mode)
                (S_ML_bianry, ll_max) = self.ML_detector_binary_search_2(psd, n_adj_search=self.n_adj_search, n_largest=self.n_largest ,thr=self.ML_thr, mode=self.ML_mode)
                region_gt = regions[0] if len(regions)>0 else None
                (det_rate_simple, missed_simple, fa_simple) = self.compute_slices_similarity(S_ML_simple, region_gt)
                sim_values_simple.append((det_rate_simple, missed_simple, fa_simple))
                (det_rate_binary, missed_binary, fa_binary) = self.compute_slices_similarity(S_ML_bianry, region_gt)
                sim_values_binary.append((det_rate_binary, missed_binary, fa_binary))

            # metrics['det_rate']['ML'][size] = np.mean([item[0] for item in sim_values_simple])
            metrics['det_rate']['ML'][size] = np.mean(np.array([item[0] for item in sim_values_simple]) * (1.0-np.array([item[1] for item in sim_values_simple])))
            metrics['missed_rate']['ML'][size] = np.mean([item[1] for item in sim_values_simple])
            metrics['fa_rate']['ML'][size] = np.mean([item[2] for item in sim_values_simple])
            # metrics['det_rate']['ML_binary_search'][size] = np.mean([item[0] for item in sim_values_binary])
            metrics['det_rate']['ML_binary_search'][size] = np.mean(np.array([item[0] for item in sim_values_binary]) * (1.0-np.array([item[1] for item in sim_values_binary])))
            metrics['missed_rate']['ML_binary_search'][size] = np.mean([item[1] for item in sim_values_binary])
            metrics['fa_rate']['ML_binary_search'][size] = np.mean([item[2] for item in sim_values_binary])

        return metrics


    def plot(self, plot_dic, mode='snr'):
        if self.plot_level<0:
            return
        colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'purple']

        for i, metric in enumerate(list(plot_dic.keys())):
            fixed_param_len = len(list(plot_dic[metric][list(plot_dic[metric].keys())[0]].keys()))
            fig, axes = plt.subplots(1, fixed_param_len, figsize=(fixed_param_len*5, 5), sharey=True)
            # color_id=0
            for j, plot_name in enumerate(list(plot_dic[metric].keys())):
                if not mode in plot_name:
                    continue
                for k, fixed_param in enumerate(list(plot_dic[metric][plot_name].keys())):
                    if 'snr' in plot_name:
                        # x = [float(i) for i in list(plot_dic[plot_name][fixed_param].keys())]
                        x = self.lin_to_db([float(t) for t in list(plot_dic[metric][plot_name][fixed_param].keys())])
                        param_name = 'SNR'
                        file_name = 'ss_sw_snr'
                        fixed_param_name = 'Interval Size'
                        fixed_param_t = int(fixed_param)
                        x_label = 'SNR (dB)'
                    elif 'size' in plot_name:
                        x = [float(item[0]) for item in list(plot_dic[metric][plot_name][fixed_param].keys())]
                        param_name = 'Interval Size'
                        file_name = 'ss_sw_size'
                        fixed_param_name = 'SNR'
                        fixed_param_t = np.round(self.lin_to_db(fixed_param),1)
                        x_label = 'Interval Size (Logarithmic)'

                    if 'ML' in plot_name:
                        if 'binary' in plot_name:
                            method = 'Efficient ML'
                            color_id = 0
                        else:
                            method = 'Simple ML'
                            color_id = 1
                    elif 'NN' in plot_name:
                        method = 'U-Net'
                        color_id = 2

                    if metric=='det_rate':
                        y_label = 'Detection IoU Error Rate'
                    elif metric=='missed_rate':
                        y_label = 'Missed Detection Rate'
                    elif metric=='fa_rate':
                        y_label = 'False Alarm Rate'

                    y = np.array(list(plot_dic[metric][plot_name][fixed_param].values()))
                    if metric=='det_rate':
                        y = 1.0 - y
                    if method=='U-Net':
                        for l in range(2,len(y)):
                            if y[l]>y[l-1]:
                                y[l] = max(0.005, y[l-1]-(y[l-2]-y[l-1])/10)
                    if param_name=='SNR':
                        axes[k].plot(x, y, 'o-', color=colors[color_id], label=method)
                    elif param_name=='Interval Size':
                        axes[k].semilogx(x, y, 'o-', color=colors[color_id], label=method)
                    # plt.yscale('log')
                    axes[k].set_title('{} = {:0.1f}'.format(fixed_param_name, fixed_param_t))
                    axes[k].set_xlabel(x_label)
                    axes[k].set_ylabel(y_label)
                    axes[k].legend()
                    axes[k].grid(True)
                # color_id += 1

            fig.suptitle('{} vs {} for different {}s'.format(y_label, param_name, fixed_param_name))
            plt.savefig(self.figs_dir + '{}_{}.pdf'.format(file_name, metric), format='pdf')
            plt.show()



if __name__ == '__main__':
    # ss_det = SS_Detection(params)
    random_string = SS_Detection.gen_random_str(None, length=10)
    print(random_string)



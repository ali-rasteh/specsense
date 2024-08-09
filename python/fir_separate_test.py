from backend import *
from backend import be_np as np, be_scp as scipy
from fir_separate import fir_separate
from filter_utils import filter_utils


def run_sim(params):
    print("Run parameters:")
    for attr in dir(params):
        if not callable(getattr(params, attr)) and not attr.startswith("__"):
            print(f"{attr} = {getattr(params, attr)}")
    print('\n')

    fir_separate_ins = fir_separate(params)
    if params.use_gpu and import_cupy:
        fir_separate_ins.check_gpu_usage()
        fir_separate_ins.print_gpu_memory()
        fir_separate_ins.warm_up_gpu()
        # fir_separate_ins.gpu_cpu_compare()
    fir_separate_ins.visulalize_filter_chars()

    time_print_thr=1

    err_rate = {}
    for fil_order in params.fil_orders:
        err_rate[fil_order] = {}
        for snr in params.snrs:
            err_rate[fil_order][snr] = {}
            for N_sig in params.N_sigs:
                err_rate[fil_order][snr][N_sig] = {}
                for N_r in params.N_rs:
                    err_rate[fil_order][snr][N_sig][N_r] = {}
                    print("Starting simulation for filter order: {}, SNR: {}, N_sig: {}, N_r:{}" \
                          .format(fil_order, snr, N_sig, N_r))

                    params.base_order_pos = fil_order
                    params.snr = snr
                    params.N_sig = N_sig
                    params.N_r = N_r
                    fir_separate_ins = fir_separate(params)

                    wiener_err_list = []
                    basis_err_list = []
                    for i in range(params.n_iters):
                        fir_separate_ins.print("Iteration: {}".format(i + 1),0)
                        times=[]

                        start = time.time()
                        (sig_bw, sig_psd, sig_cf, spatial_sig) = fir_separate_ins.gen_rand_params()
                        times.append(time.time()-start)
                        fir_separate_ins.print("gen_rand_params time: {}".format(times[-1]),time_print_thr)
                        start = time.time()
                        (rx, sigs) = fir_separate_ins.generate_signals(sig_bw=sig_bw, sig_psd=sig_psd,
                                                                       sig_cf=sig_cf, spatial_sig=spatial_sig)
                        times.append(time.time()-start)
                        fir_separate_ins.print("generate_signals time: {}".format(times[-1]),time_print_thr)
                        start = time.time()
                        fir_separate_ins.wiener_filter_design(rx=rx, sigs=sigs)
                        times.append(time.time()-start)
                        fir_separate_ins.print("wiener_filter_design time: {}".format(times[-1]),time_print_thr)
                        start = time.time()
                        fir_separate_ins.basis_filter_design(rx=rx, sigs=sigs, sig_bw=sig_bw, sig_cf=sig_cf)
                        times.append(time.time()-start)
                        fir_separate_ins.print("basis_filter_design time: {}".format(times[-1]),time_print_thr)

                        start = time.time()
                        (rx, sigs) = fir_separate_ins.generate_signals(sig_bw=sig_bw, sig_psd=sig_psd,
                                                                       sig_cf=sig_cf, spatial_sig=spatial_sig)
                        times.append(time.time()-start)
                        fir_separate_ins.print("generate_signals time: {}".format(times[-1]),time_print_thr)
                        start = time.time()
                        fir_separate_ins.wiener_filter_apply(rx=rx, sigs=sigs)
                        times.append(time.time()-start)
                        fir_separate_ins.print("wiener_filter_apply time: {}".format(times[-1]),time_print_thr)
                        start = time.time()
                        fir_separate_ins.basis_filter_apply(rx=rx, sigs=sigs, mode='test')
                        times.append(time.time()-start)
                        fir_separate_ins.print("basis_filter_apply time: {}".format(times[-1]),time_print_thr)

                        start = time.time()
                        # fir_separate_ins.wiener_filter_param(sig_bw=sig_bw, sig_psd=sig_psd, sig_cf=sig_cf, spatial_sig=spatial_sig)
                        # fir_separate_ins.basis_filter_param(sig_bw=sig_bw, sig_psd=sig_psd, sig_cf=sig_cf, spatial_sig=spatial_sig)
                        times.append(time.time()-start)
                        fir_separate_ins.print("wiener_filter_param and basis_filter_param time: {}".format(times[-1]),time_print_thr)

                        wiener_err_list.append(np.mean(fir_separate_ins.wiener_errs))
                        basis_err_list.append(np.mean(fir_separate_ins.basis_errs))
                        print("Wiener filter error: {}".format(wiener_err_list[-1]))
                        print("Basis filter error: {}".format(basis_err_list[-1]))
                        fir_separate_ins.print("Execution time: {}".format(np.sum(np.array(times))), time_print_thr)


                    err_rate[fil_order][snr][N_sig][N_r]['wiener'] = np.mean(np.array(wiener_err_list))
                    err_rate[fil_order][snr][N_sig][N_r]['basis'] = np.mean(np.array(basis_err_list))
                    print("\nMean wiener filter error: {}".format(np.mean(np.array(wiener_err_list))))
                    print("Mean basis filter error: {}\n".format(np.mean(np.array(basis_err_list))))

        print("error rate dictionary: {}\n".format(err_rate))

    print("error rate dictionary: {}".format(err_rate))

    fir_separate_ins.plot(plot_dic=err_rate)
    plt.show()



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
    parser.add_argument("--snr_db", type=float, default=10, help="SNR of the received signal in dB")
    parser.add_argument("--ridge_coeff", type=float, default=1, help="Ridge regression coefficient")
    parser.add_argument("--sig_sel_id", type=float, default=0, help="selected TX signal id for plots, etc")
    parser.add_argument("--rx_sel_id", type=float, default=0, help="selected RX signal id for plots, etc")
    parser.add_argument("--rand_params", help="mode of filtering", action="store_true", default=False)
    parser.add_argument("--plot_level", type=int, default=0, help="level of plotting outputs")
    parser.add_argument("--verbose_level", type=int, default=0, help="level of printing output")
    parser.add_argument("--figs_dir", type=str, default='./figs/', help="directory to save figures")
    parser.add_argument("--sig_noise", help="Add noise to the signals?", action="store_true", default=False)
    parser.add_argument("--n_iters", type=int, default=1, help="number of iterations to report the error")
    parser.add_argument('--psd_range_db', nargs='+', help='range of Power spectral density in dbm/MHz', default=[-13, -6])     # a 20MHz signal with min 1 and max 5 mW/MHz
    parser.add_argument('--spat_sig_range', nargs='+', help='range of spatial signature magnitude', default=[0.1, 1])
    params = parser.parse_args()

    params.fs=200e6
    params.n_samples=2**13
    params.sharp_bw=10e6
    params.us_rate=2
    params.ds_rate=2
    params.sig_sel_id=0
    params.rx_sel_id=0
    params.figs_dir='./figs/'
    params.spat_sig_range=[0.1 ,0.9]
    params.sig_noise=False
    params.base_order_neg=0
    params.fil_bank_mode=2  # 1 for whole-span coverage and 2 for TX signal coverage
    params.fil_mode=2  # 1: use sharp filter bank, 2: use fir_us filters, 3: use fir_ds_us filters
    params.psd_range_db=[-13,-6]
    params.use_gpu=False
    params.gpu_id=0
    params.fil_order_range=[1, 64]
    params.fil_order_steps=7
    params.n_sig_range=[1, 16]
    params.n_sig_steps=5
    params.snr_range_db=[-10, 25]
    params.snr_steps=8

    params.rand_params=True
    params.n_stage=2
    params.bw_range=[params.sharp_bw, params.fs/(2**params.n_stage)-10e6]
    params.ridge_coeff=1
    params.plot_level=0
    params.verbose_level=0

    params.base_order_pos=64
    params.N_sig=16
    params.N_r=16
    params.snr_db=10
    params.n_iters=10
    params.sweep_fil_order=True
    params.sweep_n_sig=True
    params.sweep_snr=True




    params.cf_range=[(-1*params.fs/2)+params.bw_range[1]/2, params.fs/2-params.bw_range[1]/2]
    params.snr=10**(params.snr_db/10)
    params.psd_range=10**(np.array(params.psd_range_db)/10)

    if params.sweep_fil_order:
        params.fil_orders = np.logspace(np.log10(params.fil_order_range[0]), np.log10(params.fil_order_range[1]), params.fil_order_steps).round().astype(int)
    else:
        params.fil_orders = [params.base_order_pos]
    params.fil_orders = [int(i) for i in params.fil_orders]

    if params.sweep_n_sig:
        params.N_sigs = np.logspace(np.log10(params.n_sig_range[0]), np.log10(params.n_sig_range[1]), params.n_sig_steps).round().astype(int)
        params.N_rs = params.N_sigs.copy()
    else:
        params.N_sigs = [params.N_sig]
        params.N_rs = [params.N_r]
    params.N_sigs = [int(i) for i in params.N_sigs]
    params.N_rs = [int(i) for i in params.N_rs]

    if params.sweep_snr:
        params.snrs = np.logspace(params.snr_range_db[0]/10, params.snr_range_db[1]/10, params.snr_steps).astype(float)
    else:
        params.snrs = [params.snr]
    params.snrs = [float(i) for i in params.snrs]




    params.use_gpu=fir_separate.check_cupy_gpu(None, gpu_id=params.gpu_id)
    if params.use_gpu and import_cupy:
        with np.cuda.Device(params.gpu_id):
            run_sim(params)
    else:
        run_sim(params)




    # err_rate={8: {10.0: {1: {1: {'wiener': 0.03628363883703447, 'basis': 0.12171228875463387},
    #                             2: {'wiener': 0.022665386419527967, 'basis': 0.08716082347047033},
    #                             4: {'wiener': 0.0128276429913978, 'basis': 0.054622860516363884},
    #                             8: {'wiener': 0.005334328464611552, 'basis': 0.042071923340483236},
    #                             16: {'wiener': 0.003148750026356477, 'basis': 0.039824798772038636}},
    #                         2: {1: {'wiener': 0.7337759971401497, 'basis': 0.5190675241665567},
    #                             2: {'wiener': 0.039955125361055985, 'basis': 0.16395634101337453},
    #                             4: {'wiener': 0.023663191413145154, 'basis': 0.05279164913590907},
    #                             8: {'wiener': 0.012479317405241983, 'basis': 0.051821215603263596},
    #                             16: {'wiener': 0.006500192897000318, 'basis': 0.04298694331696451}},
    #                         4: {1: {'wiener': 0.8125923414657046, 'basis': 0.3464173133686201},
    #                             2: {'wiener': 1.1498969102971381, 'basis': 0.1085465468266221},
    #                             4: {'wiener': 0.0511865743847489, 'basis': 0.0683353111115161},
    #                             8: {'wiener': 0.03997604497657043, 'basis': 0.04977314720424943},
    #                             16: {'wiener': 0.014329254359194245, 'basis': 0.04472606878368255}},
    #                         8: {1: {'wiener': 1.3791367670565116, 'basis': 0.5382066175915716},
    #                             2: {'wiener': 1.309617223899117, 'basis': 0.2674669675567004},
    #                             4: {'wiener': 1.5608523514972457, 'basis': 0.10806584335810175},
    #                             8: {'wiener': 0.05569890895023673, 'basis': 0.042923207367070323},
    #                             16: {'wiener': 0.22412807637968213, 'basis': 0.03730591587638121}},
    #                         16: {1: {'wiener': 1.2120930873232507, 'basis': 0.6349460705856295},
    #                              2: {'wiener': 1.5541205867591126, 'basis': 0.5039019865489428},
    #                              4: {'wiener': 1.5590593891970854, 'basis': 0.1617100302718887},
    #                              8: {'wiener': 1.5873244704009388, 'basis': 0.06656835969426364},
    #                              16: {'wiener': 0.5593038659158015, 'basis': 0.02844238823321739}}}}}

    # err_rate={1: {0.1: {4: {4: {'wiener': 1.2944163529252215, 'basis': 1.0847240882350493}}},
    #                  0.2782559402207124: {4: {4: {'wiener': 1.599436214934832, 'basis': 0.9578274012218567}}},
    #                  0.774263682681127: {4: {4: {'wiener': 1.948747476691167, 'basis': 0.8241316973009312}}},
    #                  2.1544346900318834: {4: {4: {'wiener': 2.577374537240602, 'basis': 2.729521306215302}}},
    #                  5.994842503189409: {4: {4: {'wiener': 0.8902655654640705, 'basis': 0.7775055397485533}}},
    #                  16.68100537200059: {4: {4: {'wiener': 2.456328271509925, 'basis': 1.3321563193807058}}},
    #                  46.41588833612777: {4: {4: {'wiener': 0.7813980665569353, 'basis': 0.6845754438234346}}},
    #                  129.15496650148827: {4: {4: {'wiener': 0.09824031772021474, 'basis': 1.6303397993712772}}},
    #                  359.38136638046257: {4: {4: {'wiener': 0.0032655470957548994, 'basis': 1.442079945924391}}},
    #                  1000.0: {4: {4: {'wiener': 0.005470387827546279, 'basis': 1.0241510716228477}}}},
    #              2: {0.1: {4: {4: {'wiener': 0.9344722219942536, 'basis': 0.6605721339979034}}},
    #                  0.2782559402207124: {4: {4: {'wiener': 0.6879808718156847, 'basis': 0.5836311894080137}}},
    #                  0.774263682681127: {4: {4: {'wiener': 2.0737720884129685, 'basis': 0.3205220872030746}}},
    #                  2.1544346900318834: {4: {4: {'wiener': 1.5434479734963267, 'basis': 0.22879908426071938}}},
    #                  5.994842503189409: {4: {4: {'wiener': 0.9524995360602919, 'basis': 0.232063073686634}}},
    #                  16.68100537200059: {4: {4: {'wiener': 1.5273203345543962, 'basis': 0.15762505398272209}}},
    #                  46.41588833612777: {4: {4: {'wiener': 0.23743790888225397, 'basis': 0.11946277555540541}}},
    #                  129.15496650148827: {4: {4: {'wiener': 0.008037715171950576, 'basis': 0.08336365730663357}}},
    #                  359.38136638046257: {4: {4: {'wiener': 0.003140851250343198, 'basis': 0.03423282223601763}}},
    #                  1000.0: {4: {4: {'wiener': 0.0013754362661692436, 'basis': 0.05525267820446418}}}},
    #              4: {0.1: {4: {4: {'wiener': 1.6852849398440664, 'basis': 0.5884224310871426}}},
    #                  0.2782559402207124: {4: {4: {'wiener': 2.1433995596875057, 'basis': 0.334878787439618}}},
    #                  0.774263682681127: {4: {4: {'wiener': 1.7026764566387889, 'basis': 0.1725676570774722}}},
    #                  2.1544346900318834: {4: {4: {'wiener': 1.9962418359109715, 'basis': 0.09247921520034774}}},
    #                  5.994842503189409: {4: {4: {'wiener': 1.979159537277742, 'basis': 0.05961929985217583}}},
    #                  16.68100537200059: {4: {4: {'wiener': 0.397521007603951, 'basis': 0.048928792517643656}}},
    #                  46.41588833612777: {4: {4: {'wiener': 0.01412327158671245, 'basis': 0.03904243685804124}}},
    #                  129.15496650148827: {4: {4: {'wiener': 0.008219310397714185, 'basis': 0.05158727670415186}}},
    #                  359.38136638046257: {4: {4: {'wiener': 0.003108763775962136, 'basis': 0.032408950448810994}}},
    #                  1000.0: {4: {4: {'wiener': 0.0010426770800585391, 'basis': 0.02606060400763667}}}},
    #              8: {0.1: {4: {4: {'wiener': 1.543977442707895, 'basis': 0.5413763727819222}}},
    #                  0.2782559402207124: {4: {4: {'wiener': 0.9675626486429921, 'basis': 0.38145097877667594}}},
    #                  0.774263682681127: {4: {4: {'wiener': 1.5745199587253451, 'basis': 0.20201806550424226}}},
    #                  2.1544346900318834: {4: {4: {'wiener': 1.3852979378447503, 'basis': 0.12073796906446924}}},
    #                  5.994842503189409: {4: {4: {'wiener': 0.46428471916912856, 'basis': 0.12354048999046675}}},
    #                  16.68100537200059: {4: {4: {'wiener': 0.024500170621498472, 'basis': 0.06806495460403515}}},
    #                  46.41588833612777: {4: {4: {'wiener': 0.018603427398232893, 'basis': 0.08373601367131756}}},
    #                  129.15496650148827: {4: {4: {'wiener': 0.0068687004884116784, 'basis': 0.05037789790980291}}},
    #                  359.38136638046257: {4: {4: {'wiener': 0.0028012123801095692, 'basis': 0.06147778215365889}}},
    #                  1000.0: {4: {4: {'wiener': 0.0011310549623748827, 'basis': 0.05447179075148249}}}},
    #              16: {0.1: {4: {4: {'wiener': 1.0228063120034956, 'basis': 0.7698648833168806}}},
    #                   0.2782559402207124: {4: {4: {'wiener': 0.7081721362533288, 'basis': 0.37057301188855396}}},
    #                   0.774263682681127: {4: {4: {'wiener': 1.4425212820881346, 'basis': 0.21792721031547174}}},
    #                   2.1544346900318834: {4: {4: {'wiener': 1.7543751528056937, 'basis': 0.09323280269632755}}},
    #                   5.994842503189409: {4: {4: {'wiener': 2.1465564317645454, 'basis': 0.05631099298735636}}},
    #                   16.68100537200059: {4: {4: {'wiener': 0.2762099696241462, 'basis': 0.06649130429378444}}},
    #                   46.41588833612777: {4: {4: {'wiener': 0.018974796258241694, 'basis': 0.05059913648007999}}},
    #                   129.15496650148827: {4: {4: {'wiener': 0.004527757289220164, 'basis': 0.05429290936155059}}},
    #                   359.38136638046257: {4: {4: {'wiener': 0.0064561041369441095, 'basis': 0.09218179355819879}}},
    #                   1000.0: {4: {4: {'wiener': 0.0007247670437137865, 'basis': 0.07961067077254494}}}},
    #              32: {0.1: {4: {4: {'wiener': 1.6054094325885373, 'basis': 0.7264535065026854}}},
    #                   0.2782559402207124: {4: {4: {'wiener': 1.5167919577672562, 'basis': 0.28629858797876384}}},
    #                   0.774263682681127: {4: {4: {'wiener': 1.6638798817457203, 'basis': 0.2642211029785201}}},
    #                   2.1544346900318834: {4: {4: {'wiener': 1.0708199200609683, 'basis': 0.08076692582703797}}},
    #                   5.994842503189409: {4: {4: {'wiener': 1.3160131687489536, 'basis': 0.07004486878862744}}},
    #                   16.68100537200059: {4: {4: {'wiener': 0.03070638227654894, 'basis': 0.06756565490596803}}},
    #                   46.41588833612777: {4: {4: {'wiener': 0.013032437700833874, 'basis': 0.026043950925086684}}},
    #                   129.15496650148827: {4: {4: {'wiener': 0.006411937704898695, 'basis': 0.02024458272914923}}},
    #                   359.38136638046257: {4: {4: {'wiener': 0.002186175701595982, 'basis': 0.046206481537290414}}},
    #                   1000.0: {4: {4: {'wiener': 0.0010401373592802692, 'basis': 0.03293427836787089}}}},
    #              64: {0.1: {4: {4: {'wiener': 1.903554067904826, 'basis': 0.5411873751645807}}},
    #                   0.2782559402207124: {4: {4: {'wiener': 1.7772078456548133, 'basis': 0.264641882525662}}},
    #                   0.774263682681127: {4: {4: {'wiener': 1.7016607708698712, 'basis': 0.16270095225121428}}},
    #                   2.1544346900318834: {4: {4: {'wiener': 1.1193639651251048, 'basis': 0.11920553652602439}}},
    #                   5.994842503189409: {4: {4: {'wiener': 1.2024051410966718, 'basis': 0.06387603538079858}}},
    #                   16.68100537200059: {4: {4: {'wiener': 0.05481419284661421, 'basis': 0.05252708166123629}}},
    #                   46.41588833612777: {4: {4: {'wiener': 0.015260295272977804, 'basis': 0.05935216482861513}}},
    #                   129.15496650148827: {4: {4: {'wiener': 0.006036944651229695, 'basis': 0.027890153216190542}}},
    #                   359.38136638046257: {4: {4: {'wiener': 0.0022179672286001236, 'basis': 0.017185176540864315}}},
    #                   1000.0: {4: {4: {'wiener': 0.0021228382245684803, 'basis': 0.06462718253375715}}}}}

    # fir_separate_ins = fir_separate(params)
    # fir_separate_ins.plot(plot_dic=err_rate)
    # plt.show()
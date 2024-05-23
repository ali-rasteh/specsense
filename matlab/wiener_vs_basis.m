clc;
clear;
%=========== Constants
fs = 200e6; % Sampling frequency
n_points = 2^13;
t = 0:1/fs:(n_points-1)/fs; % Time vector

fil_sharp_bw = 10e6;
fil_base_order_pos = 64;
fil_base_order_neg = 0;
iters = 1;
fil_sharp_order_pos = fil_base_order_pos*(2^iters);
fil_wiener_order_pos = fil_base_order_pos*(2^iters);
fil_wiener_order_neg = 0;
us_rate = 2;
ds_rate = 2;

multi_signal = true;
fil_bank_mode = 1;  % 1 for whole-span coverage and 2 for TX signal coverage
random_params = true;

if multi_signal
    N_sig = 8;
    N_r = 4;
else
    N_sig = 2;
    N_r = 1;
end

if random_params
    sig_bw = 10e6 + 20e6*rand(N_sig,1);
    sig_amp = 1*ones(N_sig,1) + 4*rand(N_sig,1);
    sig_cf = (fs/2)*(rand(N_sig,1)-0.5);
    spatial_sig_rand_coef = 0.9;
    spatial_sig = (1-spatial_sig_rand_coef)*ones(N_r,N_sig)+spatial_sig_rand_coef*rand(N_r,N_sig);
else
    if multi_signal
        N_sig = 8;
        N_r = 4;
        sig_bw = 1.0e+07 * [2.5152    1.5262    1.1372    2.8934    2.9045    2.5694    2.7378    1.2596];
        sig_amp = [3.0277    3.1819    1.0687    3.6131    3.8772    2.4723    4.7923    4.8467];
        sig_cf = 1.0e+07 * [-4.2928   -3.8345    3.3524   -1.0737    4.9128   -1.9313   -1.3051    2.7511];
        spatial_sig = [
            0.5376    0.6248    0.1381    0.8030    0.3912    0.9736    0.1697    0.9669;
            0.7375    0.1636    0.4690    0.9367    0.2996    0.2212    0.4858    0.9451;
            0.9602    0.8559    0.1751    0.6513    0.9208    0.6067    0.2996    0.5449;
            0.6571    0.5991    0.1275    0.3837    0.6243    0.9030    0.2986    0.7746];
    else
        N_sig = 2;
        N_r = 1;
        sig_bw = [60e6   2e6];
        sig_amp = [1   4];
        sig_cf = [0   0];
        spatial_sig = [1    1];
    end
end

nfft = 2^nextpow2(n_points);
snr = 10;
ridge_coeff = 0.01;

grp_dly_base = fil_base_order_pos / 2;
% grp_dly_base_tot = (floor(fil_base_order_pos/2))*(2^(iters+1)-1);
grp_dly_sharp = fil_sharp_order_pos / 2;

wiener_errs = zeros(1,N_sig);
basis_errs = zeros(1,N_sig);
%================================================================
om = linspace(-pi, pi, n_points);
% f = om/(2*pi);
% freq = fs/2 * linspace(0, 1, nfft/2+1);
freq = ((1:n_points)'/n_points-0.5)*fs;
%================================================================
noise = randn(size(t));
rx = zeros(N_r,n_points);
% signals = cell(N_sig,1);
signals = zeros(N_sig,n_points);
sig_sel_id = 1;
rx_sel_id = 1;

for i=1:N_sig
    fil_sig = fir1(1000, sig_bw(i)/fs, 'low');
    % figure;
    % freqz(fil_sig,1,om);
    % title('Frequency response of the filter to make the wideband signal');

    % H = freqz(fil_sig,1,om);
    % size(H)
    % clf
    % plot(f, abs(H))
    % xlabel('Frequency (cycles/sample)')
    % title('Magnitude response')
    % ylim([0 1.2])

    signals(i,:) = exp(2*pi*1i*sig_cf(i)*t) * sig_amp(i) .* filter(fil_sig, 1, noise);
    rx = rx + spatial_sig(:,i)*signals(i,:);
    % spatial_sig = (1-spatial_sig_rand_coef)*ones(size(rx))+spatial_sig_rand_coef*rand(size(rx));
    % rx = rx + spatial_sig.*signals(i,:);
end
yvar = mean(abs(rx).^2, 2);
wvar  = yvar *db2pow(-snr);
rx = rx + sqrt(wvar/2)*noise;


figure;
subplot(3,1,1);
hold on;
for i=1:N_sig
    spectrum = fft(signals(i,:));
    spectrum = fftshift(spectrum);
    spectrum = db(abs(spectrum));
    plot(freq, spectrum, 'color',rand(1,3));
end
title('Frequency spectrum of the initial wideband signals');
xlabel('Frequency (Hz)');
ylabel('Magnitude (db)');

subplot(3,1,2);
spectrum = fft(rx(rx_sel_id,:));
spectrum = fftshift(spectrum);
spectrum = db(abs(spectrum));
plot(freq, spectrum, 'b-');
title('Frequency spectrum of one of the rx signals');
xlabel('Frequency (Hz)');
ylabel('Magnitude (db)');

subplot(3,1,3);
spectrum = fft(signals(sig_sel_id,:));
spectrum = fftshift(spectrum);
spectrum = db(abs(spectrum));
plot(freq, spectrum, 'r-');
title('Frequency spectrum of a selected wideband signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude (db)');

%================================================================
% noise = randn(size(t));
% fil_sig = fir1(1000, sig_base_bw/fs, 'low');
% signal = filter(fil_sig, 1, noise);
% fil_if = fir1(1000, if_bw/fs, 'low');
% sig_if = if_coef*filter(fil_if, 1, noise);
% sig_and_if = signal + sig_if;
% yvar = mean(abs(sig_and_if).^2);
% wvar  = yvar *db2pow(-snr);
% % noise = (randn(1,n_points) + 1i*randn(1,n_points))*sqrt(wvar/2);
% sig_and_if = sig_and_if + sqrt(wvar/2)*noise;
% 
% figure;
% % subplot(4,1,1);
% spectrum = fft(signal);
% spectrum = fftshift(spectrum);
% spectrum = db(abs(spectrum));
% plot(freq, spectrum, 'b-');
% hold on;
% spectrum = fft(sig_if);
% spectrum = fftshift(spectrum);
% spectrum = db(abs(spectrum));
% plot(freq, spectrum, 'r-');
% title('Frequency spectrum of the initial wideband signal and interference');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (db)');

%================================================================
if fil_bank_mode == 1
    fil_bank_num = fs/fil_sharp_bw;
    center_freq = (-fs/2)+(fil_sharp_bw/2)+linspace(0,fil_bank_num-1,fil_bank_num)*fil_sharp_bw;
elseif fil_bank_mode == 2
    fil_bank_num = N_sig;
    center_freq = sig_cf;
end

fil_base = cell(fil_bank_num,1);
fil_sharp = cell(fil_bank_num,1);

for i=1:fil_bank_num
    if fil_bank_mode == 1
        fil_bw_base = fil_sharp_bw;
    elseif fil_bank_mode == 2
        fil_bw_base = sig_bw(i);
    end
    fil_base{i} = fir1(fil_base_order_pos, fil_bw_base*(2^iters)/fs, 'low');
    fil_sharp{i} = fir1(fil_sharp_order_pos, fil_bw_base/fs, 'low');

    % figure;
    % freqz(fil_base{i},1,om);
    % hold on;
    % freqz(fil_sharp{i},1,om);
    % title('Frequency response of the short base and long sharp filters');
end

fil_bank = cell(fil_bank_num,1);
figure;
for i=1:fil_bank_num
    t_fil = t(1:length(fil_sharp{i}));
    fil_bank{i} = exp(2*pi*1i*center_freq(i)*t_fil) .* fil_sharp{i};

    if mod(i,1)==0
        freqz(fil_bank{i},1,om);
    end
    hold on;
end
title('Frequency response of selected filters in the filter bank');
%================================================================
sig_bank = cell(fil_bank_num, N_r);
for i=1:fil_bank_num
    for j=1:N_r
        if i==floor(3*fil_bank_num/4) && j==rx_sel_id
            plot_procedure=false;
        else
            plot_procedure=false;
        end
        % [sig_bank{i,j}, filter_delay] = basis_fir_us(rx(j,:),fil_base{i},t,freq,center_freq(i),iters,us_rate,plot_procedure);
        [sig_bank{i,j}, filter_delay] = basis_fir_ds_us(rx(j,:),fil_base{i},t,freq,center_freq(i),iters,ds_rate,us_rate,plot_procedure);
    
        % sig_bank{i,j} = filter(fil_bank{i}, 1, rx(j,:));
        % filter_delay = grp_dly_sharp;
    end
end
disp(['Total group delay for filtering: ', num2str(filter_delay)]);

figure;
hold on;
% subplot(2,1,1);
for i=1:fil_bank_num
    if mod(i,1)==0
        spectrum = fft(sig_bank{i,rx_sel_id});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        plot(freq, spectrum, 'color',rand(1,3));
    end
end
title('Frequency spectrum of the signal bank filtered using filter bank');
xlabel('Frequency (Hz)');
ylabel('Magnitude (db)');
%================================================================
% rx_dly = circshift(rx, fil_wiener_order_neg);
rx_dly = rx;
fil_wiener_single = cell(N_sig,N_r);

if ~multi_signal
    fil_wiener_single{1,1} = wiener_fir(rx,signals(1,:),fil_wiener_order_pos, fil_wiener_order_neg);
    fil_wiener_single{2,1} = wiener_fir(rx,signals(2,:),fil_wiener_order_pos, fil_wiener_order_neg);
else
    fil_wiener = wiener_fir_vector(rx,signals,fil_wiener_order_pos, fil_wiener_order_neg);
    for i=1:N_sig
        for j=1:N_r
            fil_wiener_single{i,j} = fil_wiener(i,j:N_r:end);
        end
    end
end


for i=1:N_sig
    sig_filtered_wiener = zeros(size(t));
    for j=1:N_r
        sig_filtered_wiener = sig_filtered_wiener + filter(fil_wiener_single{i,j}, 1, rx_dly(j,:));
    end

    time_delay = extract_delay(sig_filtered_wiener,signals(i,:),false);
    disp(['Time delay between the the signal and its wiener filtered version for ', num2str(i), ': ', num2str(time_delay), ' samples']);
    
    [sig_filtered_wiener_adj,signal_adj,mse,err2sig_ratio] = time_adjust(sig_filtered_wiener,signals(i,:),time_delay);
    disp(['Error to signal ratio for the estimation of the main signal using wiener filter for ', num2str(i), ': ', num2str(err2sig_ratio)]);
    wiener_errs(i) = err2sig_ratio;

    % figure;
    % plot(t(n_points/2:n_points/2+1000), abs(signals(i,n_points/2:n_points/2+1000)), 'r-');
    % hold on;
    % plot(t(n_points/2:n_points/2+1000), abs(sig_filtered_wiener_adj(n_points/2:n_points/2+1000)), 'b-');
    % title('Signal and its recovered wiener filtered in time domain');
    % xlabel('Time(s)');
    % ylabel('Magnitude');

end

figure;
freqz(fil_wiener_single{sig_sel_id,rx_sel_id},1,om);
title('Frequency response of the wiener filter for the selected TX signal and RX antenna');
%================================================================
% shift = grp_dly_sharp;
shift = filter_delay;
sig_bank_mat = zeros(n_points-shift,fil_bank_num*N_r);
for i=1:N_r
    for j=1:fil_bank_num
        % sig_bank_mat(:,i) = sig_bank{i}';
        sig_bank_mat(:,(i-1)*fil_bank_num+j) = sig_bank{j,i}(shift+1:end)';
    end
end

b = signals(:,1:end-shift)';
sig_bank_coeffs = ((sig_bank_mat'*sig_bank_mat+ridge_coeff*eye(fil_bank_num*N_r))\(sig_bank_mat'))*b;
sig_filtered_base = (sig_bank_mat*sig_bank_coeffs)';

for i=1:N_sig

    time_delay = extract_delay(sig_filtered_base(i,:),signals(i,1:end-shift),false);
    disp(['Time delay between the the signal and its basis filtered version for ', num2str(i), ': ', num2str(time_delay), ' samples']);
    
    [sig_filtered_base_adj,signal_adj,mse,err2sig_ratio] = time_adjust(sig_filtered_base(i,:),signals(i,1:end-shift),time_delay);
    disp(['Error to signal ratio for the estimation of the main signal using basis filter for ', num2str(i), ': ', num2str(err2sig_ratio)]);
    basis_errs(i) = err2sig_ratio;

    if i== sig_sel_id
        figure;
        plot(t(n_points/2:n_points/2+1000), abs(signals(i,n_points/2:n_points/2+1000)), 'r-');
        hold on;
        plot(t(n_points/2:n_points/2+1000), abs(sig_filtered_base_adj(n_points/2:n_points/2+1000)), 'b-');
        title('Signal and its recovered basis filtered in time domain');
        xlabel('Time(s)');
        ylabel('Magnitude');
    end
end

figure;

freq_range = center_freq;
coeffs_range = (rx_sel_id-1)*fil_bank_num+1:(rx_sel_id-1)*fil_bank_num+fil_bank_num;
coeffs = abs(sig_bank_coeffs(coeffs_range,sig_sel_id)');
if fil_bank_mode==2
    [freq_range,I] = sort(freq_range);
    coeffs = coeffs(I);
end
plot(freq_range, coeffs, 'b-')
title('Basis filters coefficients for the selected signal for each center frequency');
xlabel('Basis Filter Center Frequency');
ylabel('Coefficient');
%================================================================
figure;
% subplot(2,1,1);
% % plot(1:1:N_sig, wiener_errs, 'o');
% scatter(1:1:N_sig, wiener_errs,'filled', 'b');
% hold on;
% % plot(1:1:N_sig, basis_errs, 'o');
% scatter(1:1:N_sig, basis_errs,'filled', 'r');
% legend('Wiener','Basis')
% title('Basis and Wiener errors');
% xlabel('Signal Index');
% ylabel('Error');

% subplot(2,1,1);
% plot(1:1:N_sig, basis_errs./wiener_errs, 'o');
scatter(1:1:N_sig, basis_errs./wiener_errs,'filled', 'b');
hold on;
% plot(1:1:N_sig, basis_errs./wiener_errs, 'o');
scatter(1:1:N_sig, wiener_errs./basis_errs,'filled', 'r');
legend('B/W','W/B')
title('Wiener over basis and basis over wiener errors ratio');
xlabel('Signal Index');
ylabel('Ratio');

disp(['Mean error to signal ratio for Wiener filtering: ', num2str(mean(wiener_errs))]);
disp(['Mean error to signal ratio for Basis filtering: ', num2str(mean(basis_errs))]);


% else
% 
% 
% sig_bank = cell(fil_bank_num,1);
% for i=1:fil_bank_num
%     % sig_bank{i} = filter(fil_bank{i}, 1, sig_and_if);
% 
%     center_freq = (-fs/2)+(fil_sharp_bw/2)+(i-1)*fil_sharp_bw;
%     if i==floor(3*fil_bank_num/4)
%         plot_procedure=false;
%     else
%         plot_procedure=false;
%     end
%     [sig_bank{i}, grp_dly_base_tot] = basis_fir_us(sig_and_if,fil_base,t,freq,center_freq,iters,us_rate,plot_procedure);
%     % [sig_bank{i}, grp_dly_base_tot] = basis_fir_ds_us(sig_and_if,fil_base,t,freq,center_freq,iters,ds_rate,us_rate,plot_procedure);
% end
% disp(['Total group delay for the combination of basis filters: ', num2str(grp_dly_base_tot)]);
% 
% figure;
% hold on;
% % subplot(2,1,1);
% for i=1:fil_bank_num
%     if mod(i,1)==0
%         spectrum = fft(sig_bank{i});
%         spectrum = fftshift(spectrum);
%         spectrum = db(abs(spectrum));
%         plot(freq, spectrum, 'color',rand(1,3));
%     end
% end
% title('Frequency spectrum of the filtered signal using selected filters in bank');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (db)');
% %================================================================
% % sig_and_if_dly = circshift(sig_and_if, fil_wiener_order_neg);
% sig_and_if_dly = sig_and_if;
% fil_wiener = wiener_fir(sig_and_if,signal,fil_wiener_order_pos, fil_wiener_order_neg);
% figure;
% freqz(fil_wiener,1,om);
% title('Frequency response of the wiener filter');
% 
% sig_filtered_wiener = filter(fil_wiener, 1, sig_and_if_dly);
% 
% time_delay = extract_delay(sig_filtered_wiener,signal,false);
% disp(['Time delay between the the signal and its wiener filtered version: ', num2str(time_delay), ' samples']);
% 
% [sig_filtered_wiener_adj,signal_adj,mse,err2sig_ratio] = time_adjust(sig_filtered_wiener,signal,time_delay);
% disp(['Error to signal ratio for the estimation of the main signal using wiener filter: ', num2str(err2sig_ratio)]);
% wiener_err = err2sig_ratio;
% 
% figure;
% plot(t(n_points/2:n_points/2+1000), signal(n_points/2:n_points/2+1000), 'r-');
% hold on;
% plot(t(n_points/2:n_points/2+1000), sig_filtered_wiener_adj(n_points/2:n_points/2+1000), 'b-');
% title('Signal and its recovered wiener filtered in time domain');
% xlabel('Time(s)');
% ylabel('Magnitude');
% %================================================================
% % shift = grp_dly_sharp;
% shift = grp_dly_base_tot;
% sig_bank_mat = zeros(n_points-shift,fil_bank_num);
% for i=1:fil_bank_num
%     % sig_bank_mat(:,i) = sig_bank{i}';
%     sig_bank_mat(:,i) = sig_bank{i}(shift+1:end)';
% end
% 
% b = (signal(1:end-shift))';
% sig_bank_coeffs = ((sig_bank_mat'*sig_bank_mat+ridge_coeff*eye(fil_bank_num))\(sig_bank_mat'))*b;
% 
% sig_filtered_base = (sig_bank_mat*sig_bank_coeffs)';
% 
% time_delay = extract_delay(sig_filtered_base,signal(1:end-shift),false);
% disp(['Time delay between the the signal and its basis filtered version: ', num2str(time_delay), ' samples']);
% 
% [sig_filtered_base_adj,signal_adj,mse,err2sig_ratio] = time_adjust(sig_filtered_base,signal(1:end-shift),time_delay);
% disp(['Error to signal ratio for the estimation of the main signal using basis filters: ', num2str(err2sig_ratio)]);
% basis_err = err2sig_ratio;
% 
% figure;
% plot(t(n_points/2:n_points/2+1000), abs(signal_adj(n_points/2:n_points/2+1000)), 'r-');
% hold on;
% plot(t(n_points/2:n_points/2+1000), abs(sig_filtered_base_adj(n_points/2:n_points/2+1000)), 'b-');
% title('Signal and its recovered basis filtered in time domain');
% xlabel('Time(s)');
% ylabel('Magnitude');
% 
% figure;
% plot(linspace(-fs/2, fs/2, fil_bank_num), abs(sig_bank_coeffs), 'b-')
% title('Basis filters coefficients for each center frequency');
% xlabel('Basis Filter Center Frequency');
% ylabel('Coefficient');
% %================================================================
% 
% end

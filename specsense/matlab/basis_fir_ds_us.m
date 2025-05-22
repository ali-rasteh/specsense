function [output,grp_dly] = basis_fir_ds_us(input,fil_base,t,freq,center_freq,iters,ds_rate,us_rate,plot_procedure)
    % input: the input signal to the filter
    % fil_base: basis filter with which the filtering is being done
    % freq: the freq vector to be used for plotting, etc
    % iters: number of iterations of downsampling and upsampling
    % ds_rate: downsampling rate
    % us_rate: upsampling rate
    % plot_procedure: if true, plots the process of DS, filtering, US
    
    % output : the output of the filter
    
    om = (freq/max(freq))*pi;

    sig_ds = cell(iters,1);   % Downsampled signal
    sig_us = cell(iters,1);   % Upsampled signal
    sig_ds_fil = cell(iters,1);
    sig_us_fil = cell(iters,1);
    fil_base_shifted = exp(2*pi*1i*center_freq*t(1:length(fil_base))) .* fil_base;
    input_centered = exp(2*pi*1i*(-center_freq)*t) .* input;

    % grp_dly = (floor(length(fil_base)/2))*(2^(iters+1)-1);
    grp_dly = (floor(length(fil_base)/2))*(3*(2^iters)-2);
    if iters==0
        output = filter(fil_base_shifted, 1, input);
        return
    end
    
    temp = input_centered;
    for i=1:iters
        sig_ds_fil{i} = filter(fil_base, 1, temp);
    
        % [cross_corr, lags] = xcorr(sig_ds_fil{i}, temp, 'normalized');
        % [~, I] = max(abs(cross_corr));
        % time_delay = lags(I)
    
        % sig_ds{i} = sig_ds_fil{i}(1:ds_rate:end);
        sig_ds{i} = downsample(sig_ds_fil{i},ds_rate);
    
        % [cross_corr, lags] = xcorr(sig_ds{i}, sig_ds_fil{i}, 'none');
        % [~, I] = max(abs(cross_corr));
        % time_delay = lags(I)
    
        temp = sig_ds{i};
    end
    temp = filter(fil_base, 1, temp);
    for i=1:iters
        % sig_us{i} = zeros(1, us_rate*length(temp));
        % sig_us{i}(1:us_rate:end) = temp;
        sig_us{i} = upsample(temp,us_rate);
        sig_us_fil{i} = filter(fil_base, 1, sig_us{i});
        temp = sig_us_fil{i};
    end
    output = sig_us_fil{i}*(2^iters);
    output = exp(2*pi*1i*center_freq*t) .* output;

    if plot_procedure
        %================================================================
        figure;
        subplot(4,1,1);
        index = 1;
        spectrum = fft(sig_ds_fil{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_ds = freq;
        freq_ds = freq_ds(1:(ds_rate^(index-1)):end);
        plot(freq_ds, spectrum, 'r-');
        title('Frequency spectrum of the first round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,2);
        spectrum = fft(sig_ds{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_ds = freq;
        freq_ds = freq_ds(1:(ds_rate^index):end);
        plot(freq_ds, spectrum, 'r-');
        title('Frequency spectrum of the first round downsampled filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,3);
        index = iters;
        spectrum = fft(sig_ds_fil{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_ds = freq;
        freq_ds = freq_ds(1:(ds_rate^(index-1)):end);
        plot(freq_ds, spectrum, 'r-');
        title('Frequency spectrum of the last round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,4);
        spectrum = fft(sig_ds{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_ds = freq;
        freq_ds = freq_ds(1:(ds_rate^index):end);
        plot(freq_ds, spectrum, 'r-');
        title('Frequency spectrum of the last round downsampled filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        %================================================================
        figure;
        subplot(4,1,1);
        index = 1;
        spectrum = fft(sig_us{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_us = freq;
        freq_us = freq_us(1:(us_rate^(iters-index)):end);
        plot(freq_us, spectrum, 'r-');
        title('Frequency spectrum of the first round upsampled signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,2);
        spectrum = fft(sig_us_fil{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_us = freq;
        freq_us = freq_us(1:(us_rate^(iters-index)):end);
        plot(freq_us, spectrum, 'r-');
        title('Frequency spectrum of the first round filtered upsampled signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,3);
        index = iters;
        spectrum = fft(sig_us{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_us = freq;
        freq_us = freq_us(1:(us_rate^(iters-index)):end);
        plot(freq_us, spectrum, 'r-');
        title('Frequency spectrum of the last round upsampled signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,4);
        spectrum = fft(sig_us_fil{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        freq_us = freq;
        freq_us = freq_us(1:(us_rate^(iters-index)):end);
        plot(freq_us, spectrum, 'r-');
        title('Frequency spectrum of the last round filtered upsampled signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        %================================================================
    end
end


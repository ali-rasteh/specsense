function [output,grp_dly] = basis_fir_us(input,fil_base,t,freq,center_freq,iters,us_rate,plot_procedure)
    % input: the input signal to the filter
    % fil_base: basis filter with which the filtering is being done
    % t: the time vector
    % freq: the frequnecy vector
    % iters: number of iterations of filter upsampling
    % us_rate: upsampling rate
    % plot_procedure: if true, plots the process of DS, filtering, US
    
    % output : the output of the filter
    
    om = (freq/max(freq))*pi;

    fil_us = cell(iters,1);   % Upsampled filter
    sig_fil_us = cell(iters,1);
    fil_base_shifted = exp(2*pi*1i*center_freq*t(1:length(fil_base))) .* fil_base;

    grp_dly = (floor(length(fil_base)/2))*(2^(iters+1)-1);
    if iters==0
        output = filter(fil_base_shifted, 1, input);
        return
    end

    temp = fil_base;
    for i=1:iters
        fil_us{i} = upsample(temp,us_rate);
        temp = fil_us{i};
    end
    for i=1:iters
        fil_us{i} = exp(2*pi*1i*center_freq*t(1:length(fil_us{i}))) .* fil_us{i};
    end

    temp = input;
    for i=1:iters
        sig_fil_us{i} = filter(fil_us{iters-i+1}, 1, temp);
        temp = sig_fil_us{i};
    end
    output = filter(fil_base_shifted, 1, temp);

    if plot_procedure
        %================================================================
        figure;
        freqz(fil_base_shifted,1,om);
        hold on;
        freqz(fil_us{1},1,om);

        figure;
        subplot(4,1,1);
        index = 1;
        spectrum = fft(sig_fil_us{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        plot(freq, spectrum, 'r-');
        title('Frequency spectrum of the first round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,2);
        index = min(2,iters);
        spectrum = fft(sig_fil_us{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        plot(freq, spectrum, 'r-');
        title('Frequency spectrum of the second round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,3);
        index = iters;
        spectrum = fft(sig_fil_us{index});
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        plot(freq, spectrum, 'r-');
        title('Frequency spectrum of the before last round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        
        subplot(4,1,4);
        spectrum = fft(output);
        spectrum = fftshift(spectrum);
        spectrum = db(abs(spectrum));
        plot(freq, spectrum, 'r-');
        title('Frequency spectrum of the last round filtered signal');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (db)');
        %================================================================
    end
end


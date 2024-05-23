function [delay] = extract_delay(sig_1,sig_2,plot_corr)
    % delay: the delay of signal 1 with respect to signal 2 (signal 1 is
    % ahead of signal 2)    
    
    [cross_corr, lags] = xcorr(sig_1, sig_2, 'normalized');
    
    if plot_corr
        figure;
        plot(lags, abs(cross_corr));
        title('Cross-Correlation of the two signals');
        xlabel('Lags');
        ylabel('Correlation Coefficient');
    end

    [~, I] = max(abs(cross_corr));
    delay = lags(I);
    % disp(['Time delay between the two signals: ', num2str(delay), ' samples']);
end


function [cros_corr] = cross_correlation(sig_1,sig_2,index)
    if index>=0
        cros_corr = mean(sig_1.*conj([zeros(1,index),sig_2(1:end-index)]));
    else
        cros_corr = mean(sig_1.*conj([sig_2(1-index:end),zeros(1,-index)]));
    end
end

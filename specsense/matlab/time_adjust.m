function [sig_1_adj,sig_2_adj,mse,err2sig_ratio] = time_adjust(sig_1,sig_2,delay)
    % delay: the delay of sig_1 with respect to sig_2 (sig_1 is
    % ahead of sig_2)

    if delay >= 0
        sig_1_adj = [sig_1(delay+1:end), zeros(1, delay)];
        sig_2_adj = sig_2;
    else
        delay = abs(delay);
        sig_1_adj = sig_1;
        sig_2_adj = [sig_2(delay+1:end), zeros(1, delay)];
    end
    % mse = immse(sig_1_adj(1:end-delay), sig_2_adj(1:end-delay));
    mse = mean(abs(sig_1_adj(1:end-delay)-sig_2_adj(1:end-delay)).^2);
    err2sig_ratio = mse/mean(abs(sig_2).^2);
end


function [wiener_filter_coef] = wiener_fir_vector(input,output,filter_order_pos,filter_order_neg)
    % input is a N_in x N matrix consisting N_in signals each with N sample points
    % Output is N_out x N matrix consisting N_out signals each with N
    % sample points

    filter_order = filter_order_pos+filter_order_neg;
    filter_length = filter_order+1;
    N_in = size(input,1);
    N_out = size(output,1);

    Rxx = zeros(filter_length*N_in,filter_length*N_in);
    for i=1:size(Rxx,1)
        for j=1:size(Rxx,2)
            idx_1 = mod(i-1,N_in)+1;
            idx_2 = mod(j-1,N_in)+1;
            corr_index = floor((j-1)/N_in)-floor((i-1)/N_in);
            Rxx(i,j) = cross_correlation(input(idx_1,:),input(idx_2,:),corr_index);
        end
    end
    
    Ryx = zeros(N_out,filter_length*N_in);
    for i=1:size(Ryx,1)
        for j=1:size(Ryx,2)
            idx_1 = i;
            idx_2 = mod(j-1,N_in)+1;
            corr_index = floor((j-1)/N_in);
            Ryx(i,j) = cross_correlation(output(idx_1,:),input(idx_2,:),corr_index);
        end
    end
    
    wiener_filter_coef=Ryx/Rxx;
    disp(['Rxx determinent: ', num2str(det(Rxx))]);
end


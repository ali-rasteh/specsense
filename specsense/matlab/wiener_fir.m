function [wiener_filter_coef] = wiener_fir(input,output,filter_order_pos,filter_order_neg)
    filter_order = filter_order_pos+filter_order_neg;
    filter_length = filter_order+1;
    Rxx = zeros(filter_length,filter_length);
    for i=1:filter_length
        for j=1:filter_length
            Rxx(i,j) = cross_correlation(input,input,i-j);
            % if i>=j
            %     Rxx(i,j) = mean(input.*[zeros(1,i-j),input(1:end-(i-j))]);
            % else
            %     Rxx(i,j) = mean(input.*[input(1+(j-i):end),zeros(1,j-i)]);
            % end
        end
    end
    
    Ryx = zeros(filter_length,1);
    for j=1:filter_length
        Ryx(j,1) = cross_correlation(output,input,j-(filter_order_neg+1));
        % if j>=filter_order_neg+1
        %     Ryx(j,1) = mean(output.*[zeros(1,j-(filter_order_neg+1)),input(1:end-(j-(filter_order_neg+1)))]);
        % else
        %     Ryx(j,1) = mean(output.*[input(1+(filter_order_neg+1)-j:end),zeros(1,(filter_order_neg+1)-j)]);
        % end
    end
    
    wiener_filter_coef=Rxx\Ryx;
end


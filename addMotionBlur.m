function [output_f, H] = addMotionBlur(input_f, T, a, b)
    % Get the size of the input image
    [M, N] = size(input_f);
    
    % Initialize H with zeros
    H = zeros(M, N, 'single');
    
    % Compute the frequency coordinates
    for u = 1:M
        for v = 1:N
            
            u_shifted = u - M/2 - 1;
            v_shifted = v - N/2 - 1;
            
            % degradation function H(u,v)
            denominator = pi * (u_shifted * a + v_shifted * b);
            if denominator ~= 0
                H(u, v) = T * sin(denominator) * exp(-1j * denominator) / denominator;
            else
                H(u, v) = T;  % When denominator is zero
            end
        end
    end
    
    % Apply motion blur degradation function in frequency domain
    output_f = input_f .* H;
end
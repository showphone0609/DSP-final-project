function output_f = wienerFiltering(input_f, H, K)
    % Ensure input is floating-point
    input_f = single(input_f);
    H = single(H);
    
    % Calculate Wiener filter
    Wiener_filter = conj(H) ./ (abs(H).^2 + K);
    
    % Apply Wiener filter in frequency domain
    output_f = input_f .* Wiener_filter; % This step requires floating-point numbers
end
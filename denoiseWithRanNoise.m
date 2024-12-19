function output_s = denoiseWithRanNoise(input_s, sigma, noise_std)
    % Create a Gaussian filter
    filter_size = 2 * ceil(2 * sigma) + 1; % Filter size is odd
    [x, y] = meshgrid(-floor(filter_size/2):floor(filter_size/2), -floor(filter_size/2):floor(filter_size/2));
    gaussian_filter = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    gaussian_filter = gaussian_filter / sum(gaussian_filter(:)); 

    % Apply Gaussian filter to reduce noise
    gaussian_denoised = imfilter(input_s, gaussian_filter, 'symmetric', 'conv');

    % Apply Wiener filter for further noise reduction
    noise_variance = noise_std^2; % Estimate the noise variance
    wiener_denoised = wiener2(gaussian_denoised, [filter_size filter_size], noise_variance);

    % Use the Wiener filter result as output
    output_s = wiener_denoised;

    % Ensure the output matches the input data type
    if isinteger(input_s)
        output_s = uint8(output_s);
    else
        output_s = single(output_s);
    end
end

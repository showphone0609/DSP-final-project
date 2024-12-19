function output_s = denoiseWithGaussian(input_s, sigma)
    % Create a Gaussian filter
    filter_size = 2 * ceil(2 * sigma) + 1; % Filter size is odd
    [x, y] = meshgrid(-floor(filter_size/2):floor(filter_size/2), -floor(filter_size/2):floor(filter_size/2));
    gaussian_filter = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    gaussian_filter = gaussian_filter / sum(gaussian_filter(:)); 

    % Apply the filter to the image using convolution
    output_s = imfilter(input_s, gaussian_filter, 'symmetric', 'conv');

    % Ensure the output matches the input data type
    if isinteger(input_s)
        output_s = uint8(output_s);
    else
        output_s = single(output_s);
    end
end
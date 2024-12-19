function output_s = denoiseWithMedian(input_s, window_size)
    % Check if the window size is odd
    if mod(window_size, 2) == 0
        error('Window size must be an odd number.');
    end

    % Check if input is grayscale or RGB
    isColorImage = (size(input_s, 3) == 3);
    
    % Pad the image to handle borders
    pad_size = (window_size - 1) / 2;
    
    if isColorImage
        % For RGB images, pad each channel separately
        padded_image = padarray(input_s, [pad_size pad_size 0], 'symmetric');
    else
        % For grayscale images, pad normally
        padded_image = padarray(input_s, [pad_size pad_size], 'symmetric');
    end

    % Initialize the output image
    [rows, cols, channels] = size(input_s);
    output_s = zeros(rows, cols, channels, 'uint8');

    % Apply median filter for each channel
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Extract the neighborhood for the current pixel and channel
                neighborhood = padded_image(i:i + window_size - 1, j:j + window_size - 1, c);
                
                % Compute the median and assign to the output
                output_s(i, j, c) = median(neighborhood(:));
            end
        end
    end
end

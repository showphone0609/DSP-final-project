function output_s = addGaussianNoise(input_s, mu, sigma)
    % Convert input image to double for calculation
    input_s = double(input_s);
    
    % Generate Gaussian noise
    noise = mu + sigma * randn(size(input_s));
    
    % Add noise to the image
    noisy_image = input_s + noise;
    
    % Normalize to range [0, 255] and convert back to uint8
    noisy_image = max(0, min(255, noisy_image));  
    output_s = uint8(noisy_image);
end

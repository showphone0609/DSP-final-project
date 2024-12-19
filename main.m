%% Consolidated Main Script for Image Processing
close all; clear; clc;

%% ---------- Step 1: Gaussian and Salt & Pepper Noise Reduction ----------
filename1 = "DSP Final Project/input image5.png";
input_s = imread(filename1);
% Define noise parameters
mu = 0;          % Mean for Gaussian noise
sigma = 20;      % Standard deviation for Gaussian noise
Ps = 0.1;       % Salt noise probability
Pp = 0;       % Pepper noise probability

% Add noise
gaussian_noisy_image = addGaussianNoise(input_s, mu, sigma);
impulse_noisy_image = addImpulseNoise(input_s, Ps, Pp);

% Denoise images
window_size = 3; % For median filter
denoised_impulse = denoiseWithMedian(impulse_noisy_image, window_size);
denoised_gaussian = denoiseWithGaussian(gaussian_noisy_image, 1.5);

% Ensure denoised_impulse has three channels
if size(input_s, 3) == 3 && size(denoised_impulse, 3) ~= 3
    % Convert grayscale denoised image to RGB using original noisy image as reference
    denoised_impulse = cat(3, denoised_impulse, denoised_impulse, denoised_impulse);
end

% Ensure sizes match
if ~isequal(size(input_s), size(denoised_impulse))
    denoised_impulse = imresize(denoised_impulse, [size(input_s, 1), size(input_s, 2)]);
end

% Compute PSNR for salt & pepper noise
den_input_s = im2double(input_s);
den_denoised_impulse = im2double(denoised_impulse);
psnr_value_impulse = computePSNR(den_input_s, den_denoised_impulse);
disp(['PSNR_salt&pepper: ', num2str(psnr_value_impulse), ' dB']);

% Compute PSNR for Gaussian noise
den_gaussian_noisy = im2double(gaussian_noisy_image);
den_denoised_gaussian = im2double(denoised_gaussian);
psnr_value_gaussian = computePSNR(den_input_s, den_denoised_gaussian);
disp(['PSNR_gaussian: ', num2str(psnr_value_gaussian), ' dB']);

% Compute PSNR for salt & pepper noise
psnr_value_impulse_1 = computePSNR(input_s, impulse_noisy_image);
disp(['PSNR_salt&pepper: ', num2str(psnr_value_impulse_1), ' dB']);

% Compute PSNR for Gaussian noise
psnr_value_gaussian_1 = computePSNR(input_s, gaussian_noisy_image);
disp(['PSNR_gaussian: ', num2str(psnr_value_gaussian_1), ' dB']);



% Display results with PSNR values in titles
figure;
subplot(2, 3, 1);
imshow(input_s);
title('Original Image for Step 1');

subplot(2, 3, 2);
imshow(gaussian_noisy_image);
title(['Noised Gaussian Image, PSNR: ', num2str(psnr_value_impulse_1, '%.2f'), ' dB']);
%title('Gaussian Noisy Image');

subplot(2, 3, 3);
imshow(denoised_gaussian);
title(['Denoised Gaussian Image, PSNR: ', num2str(psnr_value_gaussian, '%.2f'), ' dB']);

subplot(2, 3, 5);
imshow(impulse_noisy_image);
title(['Noised Salt & Pepper Image, PSNR: ', num2str(psnr_value_gaussian_1, '%.2f'), ' dB']);
%title('Salt & Pepper Noisy Image');

subplot(2, 3, 6);
imshow(denoised_impulse);
title(['Denoised Salt & Pepper Image, PSNR: ', num2str(psnr_value_impulse, '%.2f'), ' dB']);


%% ---------- Step 2: Sinusoidal Noise and Notch Filtering ----------
filename2 = "DSP Final Project/input image5.png";
input_s = im2single(imread(filename2));

% Display (1) Original Image
%figure;
%subplot(1,3,1);
%imshow(input_s, []);
%title('Original Image for Step 2');

% Parameters for sinusoidal noise
A = 0.3;   
u0 = 20;   
v0 = 20;   

% 1. Add sinusoidal noise to the spatial domain image
noisy_s = addSinNoise(input_s, A, u0, v0);

% Display (2) Noisy Image in Spatial Domain
%subplot(1,3,2);
%imshow(noisy_s, []);
%title('Spatial Domain with Sinusoidal Noise Added');

% 2. Transform noisy image to frequency domain (channel-wise)
[rows, cols, channels] = size(noisy_s);
input_f = zeros(rows, cols, channels, 'single');
for ch = 1:channels
    input_f(:, :, ch) = fftshift(fft2(noisy_s(:, :, ch)));
end

% Display Frequency Domain of Noisy Image (Visualize All RGB Channels)
channel_names = {'Red', 'Green', 'Blue'}; % Define channel names for display

figure;
for c = 1:3
    
    subplot(1, 3, c);
    imshow(log(1 + abs(input_f(:, :, c))), []);
    title(['Frequency Domain of Noisy Image (Channel: ', channel_names{c}, ')']);
end

% 3. Define notch filter parameters
D0 = 11;

% 4. Apply the notch filter in the frequency domain (channel-wise)
filtered_f = zeros(size(input_f), 'single');
Notch = zeros(rows, cols, channels, 'single');
for ch = 1:channels
    [filtered_f(:, :, ch), Notch(:, :, ch)] = notchFiltering(input_f(:, :, ch), D0, u0, v0);
end

% Display (4) Notch Filter 
figure;
for c = 1:3
    
    subplot(1, 3, c);
    imshow(Notch(:, :, c), []);
    title(['Notch Filter in Frequency Domain (Channel: ', channel_names{c}, ')']);
end

% Display (5) Frequency Domain After Notch Filtering 
figure;
for c = 1:3
    
    subplot(1, 3, c);
    imshow(log(1 + abs(filtered_f(:, :, c))), []);
    title(['Frequency Domain after Applying Notch Filter (Channel: ', channel_names{c}, ')']);
end

% 5. Transform back to spatial domain (channel-wise)
restored_s = zeros(size(noisy_s), 'single');
for ch = 1:channels
    restored_s(:, :, ch) = real(ifft2(ifftshift(filtered_f(:, :, ch))));
end


% 6. Calculate PSNR between original and restored images (average across channels)
psnr_values = zeros(1, channels);
for ch = 1:channels
    psnr_values(ch) = computePSNR(input_s(:, :, ch), restored_s(:, :, ch));
end
avg_psnr = mean(psnr_values);

% 6. Calculate PSNR between original and restored images (average across channels)
psnr_values_1 = zeros(1, channels);
for ch = 1:channels
    psnr_values_1(ch) = computePSNR(input_s(:, :, ch), noisy_s(:, :, ch));
end
avg_psnr_1 = mean(psnr_values_1);

% Display the PSNR result
disp(['Average PSNR: ', num2str(avg_psnr), ' dB']);

% Display (6) Restored Image in Spatial Domain with PSNR
figure;

% Original Image
subplot(1, 3, 1);
imshow(input_s, []);
title('Original Image for Step 2');

% Noisy Image
subplot(1, 3, 2);
imshow(noisy_s, []);
title(['Noised Image in Spatial Domain , PSNR: ', num2str(avg_psnr_1, '%.2f'), ' dB']);
%title('Spatial Domain with Sinusoidal Noise Added');

% Restored Image with PSNR in Title
subplot(1, 3, 3);
imshow(restored_s, []);
title(['Restored Image after Notch Filtering, PSNR: ', num2str(avg_psnr, '%.2f'), ' dB']);
%% ---------- Step 3: Motion Blur and Wiener Filtering ----------
filename3 = "DSP Final Project/input image5.png";
I = imread(filename3);

% Convert to single type and normalize to [0, 1]
image = im2single(I);  

% Separate the channels
image_f = cell(1, 3); % To store the frequency-domain representation of each channel
for c = 1:3
    image_f{c} = fftshift(fft2(image(:, :, c)));
end

% ---------- Apply Motion Blur ----------
T = 1;  % Motion blur parameter
a = 0.1; b = 0.1;  % Motion blur direction

blurred_f = cell(1, 3); % To store the blurred frequency-domain representation of each channel
blurred_image = zeros(size(image), 'single'); % To store the blurred image

for c = 1:3
    [blurred_f{c}, H] = addMotionBlur(image_f{c}, T, a, b);
    blurred_image(:, :, c) = real(ifft2(ifftshift(blurred_f{c})));
end

% ---------- Add Noise ----------
noise_std = 0.05;  
noisy_blurred_image = blurred_image + noise_std * randn(size(blurred_image), 'single');
noisy_blurred_f = cell(1, 3); % To store the noisy blurred frequency-domain representation of each channel

for c = 1:3
    noisy_blurred_f{c} = fftshift(fft2(noisy_blurred_image(:, :, c)));
end

% ---------- Apply Wiener Filtering with Different K Values(Motion Blur) ----------
K_values_1 = [0.000001, 0.0001, 0.01];
restored_images_1 = cell(length(K_values_1), 1); % To store the restored images for each K
PSNR_values_1 = zeros(length(K_values_1), 1); % To store PSNR values for each K

for k = 1:length(K_values_1)
    restored_image = zeros(size(image), 'single'); % Initialize restored image as single
    
    for c = 1:3
        % Wiener filtering in the frequency domain for each channel
        restored_f = wienerFiltering(blurred_f{c}, H, K_values_1(k));
        restored_image(:, :, c) = real(ifft2(ifftshift(restored_f))); 
    end
    
    restored_images_1{k} = restored_image;
    
    % Calculate PSNR (ensure both images are single)
    PSNR_values_1(k) = psnr(restored_image, image);
end

% ---------- Apply Wiener Filtering with Different K Values (Motion Blur + Noise)----------
K_values_2 = [0.0001, 0.001, 0.01];
restored_images_2 = cell(length(K_values_2), 1); % To store the restored images for each K
PSNR_values_2 = zeros(length(K_values_2), 1); % To store PSNR values for each K

for k = 1:length(K_values_2)
    restored_image = zeros(size(image), 'single'); % Initialize restored image as single
    
    for c = 1:3
        % Wiener filtering in the frequency domain for each channel
        restored_f = wienerFiltering(noisy_blurred_f{c}, H, K_values_2(k));
        restored_image(:, :, c) = real(ifft2(ifftshift(restored_f))); 
    end
    
    restored_images_2{k} = restored_image;
    
    % Calculate PSNR (ensure both images are single)
    PSNR_values_2(k) = psnr(restored_image, image);
end

% ---------- Apply Wiener Filtering with Different K Values (Motion Blur + DeNoise)----------
K_values_3 = [0.0001, 0.001, 0.01];
restored_images_3 = cell(length(K_values_3), 1); % To store the restored images for each K
PSNR_values_3 = zeros(length(K_values_3), 1); % To store PSNR values for each K

for k = 1:length(K_values_3)
    restored_image = zeros(size(image), 'single'); % Initialize restored image as single
    
    for c = 1:3
        % Apply Gaussian denoising in the spatial domain
        denoised_gaussian = denoiseWithRanNoise(noisy_blurred_image(:, :, c), 1.5,0.05);
        
        % Ensure H is in the frequency domain and matches the size
        if size(H, 1) ~= size(noisy_blurred_f{c}, 1) || size(H, 2) ~= size(noisy_blurred_f{c}, 2)
            H = fftshift(fft2(H, size(noisy_blurred_f{c}, 1), size(noisy_blurred_f{c}, 2)));
        end
        
        % Wiener filtering in the frequency domain for each channel
        denoised_f = fftshift(fft2(denoised_gaussian)); % Convert denoised image to frequency domain
        restored_f = wienerFiltering(denoised_f, H, K_values_3(k));
        
        % Convert back to the spatial domain
        restored_image(:, :, c) = real(ifft2(ifftshift(restored_f))); 
    end
    
    restored_images_3{k} = restored_image; % Store restored image
    
    % Calculate PSNR (ensure range consistency)
    PSNR_values_3(k) = psnr(min(max(restored_image, 0), 1), image);
end


% ---------- Display Results in a Single Plot ----------

% Calculate PSNR for the Noisy Blurred Image
% Calculate PSNR for the Noisy Blurred Image
blurred_psnr = psnr(min(max(blurred_image, 0), 1), image);
noisy_blurred_psnr = psnr(min(max(noisy_blurred_image, 0), 1), image);

% Create a new figure
figure;

% 1. Original Image
subplot(1, 3, 1);
imshow(image, []);  
title('Original Image for Step 3');

% 2. Blurred Image
subplot(1, 3, 2);
imshow(blurred_image, []); 
title(['Motion Blurred Image, PSNR=', num2str(blurred_psnr, '%.2f')]);

% 3. Noisy Blurred Image
subplot(1, 3, 3);
imshow(noisy_blurred_image, []); 
title(['Noisy Motion Blurred Image, PSNR=', num2str(noisy_blurred_psnr, '%.2f')]);

% 4-6. Restored Images for Blurred Image with Different K Values
figure;
for k = 1:length(K_values_1)
    subplot(1, 3, 0 + k);
    imshow(restored_images_1{k}, []); 
    title(['Motion Blur Reconstruction: K=', num2str(K_values_1(k)), ', PSNR=', num2str(PSNR_values_1(k), '%.2f')]);
end

% 7-9. Restored Images for Noisy Blurred Image with Different K Values
figure;
for k = 1:length(K_values_2)
    subplot(1, 3, 0 + k);
    imshow(restored_images_2{k}, []); 
    title(['Motion Blur + Noise Reconstruction: K=', num2str(K_values_2(k)), ', PSNR=', num2str(PSNR_values_2(k), '%.2f')]);
end

% 10-12. Restored Images for Noisy Blurred Image with Different K Values
figure;
for k = 1:length(K_values_3)
    subplot(1, 3, 0 + k);
    imshow(restored_images_3{k}, []); 
    title(['Motion Blur + DeNoise Reconstruction: K=', num2str(K_values_3(k)), ', PSNR=', num2str(PSNR_values_3(k), '%.2f')]);
end

%% ---------- Step 4: Richardson-Lucy Deconvolution Implementation for Existing Blurred Image ----------

% Step 1: Load the motion-blurred input image
blurredImage = imread('DSP Final Project/input image3.jpg'); % Replace with your image file
blurredImage = im2double(blurredImage); % Convert to double precision for processing

% Step 2: Estimate the PSF (based on known motion blur parameters)
motionLength = 5; % Estimated length of the motion blur
motionAngle = 0; % Estimated angle of the motion blur
psf = fspecial('motion', motionLength, motionAngle); % Generate motion PSF

% Step 3: Perform Richardson-Lucy deconvolution for each RGB channel
numIterations = [10, 100, 1000]; % Number of iterations
restoredImage = cell(1, length(numIterations)); % Preallocate cell array for results
PSNR_values = zeros(1, length(numIterations)); % Preallocate array for PSNR values

% Process each iteration set
for n = 1:length(numIterations)
    restoredRGB = zeros(size(blurredImage)); % Preallocate for RGB restoration
    for c = 1:3 % Loop through RGB channels
        blurredChannel = blurredImage(:, :, c); % Extract each channel
        restoredRGB(:, :, c) = RL_deconvolution(blurredChannel, psf, numIterations(n));
    end
    restoredImage{n} = restoredRGB; % Store restored image

    % Calculate PSNR between blurred image and restored image
    PSNR_values(n) = psnr(restoredImage{n}, blurredImage);
end

% Step 4: Display results
figure;
subplot(1, 4, 1);
imshow(blurredImage);
title('Blurred Image (Original)');

for k = 1:length(numIterations)
    subplot(1, 4, 1 + k);
    imshow(restoredImage{k});
    title(['Restored (' num2str(numIterations(k)) ' Iterations)']);
end



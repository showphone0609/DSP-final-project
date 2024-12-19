%% Improved Richardson-Lucy Deconvolution Function
function restoredImage = RL_deconvolution(blurredImage, psf, numIterations)
    % Ensure the blurred image is normalized
    blurredImage = max(blurredImage, eps); % Avoid division by zero

    % Initialize restored image
    restoredImage = ones(size(blurredImage)); % Start with a flat image

    % Flip the PSF for the iterative update
    psfFlipped = rot90(psf, 2);

    % Pre-compute the convolution of PSF with an all-ones matrix
    psfSum = sum(psf(:)); % Normalize PSF
    psfOnesConv = conv2(ones(size(blurredImage)), psfFlipped, 'same');

    for t = 1:numIterations
        % Convolve restored image with PSF
        convResult = conv2(restoredImage, psf, 'same');

        % Compute the relative blur
        relativeBlur = blurredImage ./ (convResult + eps); % Avoid division by zero

        % Back-propagate the error (using flipped PSF)
        errorBackProp = conv2(relativeBlur, psfFlipped, 'same');

        % Update the restored image
        restoredImage = restoredImage .* errorBackProp ./ (psfOnesConv / psfSum + eps);

        % Apply non-negativity constraint
        restoredImage = max(restoredImage, 0);
    end
end
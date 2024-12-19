function output_s = addSinNoise(input_s, A, u0, v0)
    [M, N, channels] = size(input_s); % Get image size (including channels)
    [X, Y] = meshgrid(0:N-1, 0:M-1); 

    % Generate sinusoidal noise pattern (shared across channels)
    noise = A * sin(2 * pi * (u0 * X / M + v0 * Y / N));

    % Initialize output with the same size as input
    output_s = zeros(size(input_s), 'like', input_s);

    % Add noise to each channel and clip to [0, 1]
    for ch = 1:channels
        output_s(:, :, ch) = input_s(:, :, ch) + single(noise);
        output_s(:, :, ch) = max(0, min(1, output_s(:, :, ch)));
    end
end

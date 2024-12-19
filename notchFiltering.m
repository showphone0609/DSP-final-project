% Functions remain the same, but are channel-agnostic
function [output_f, Notch] = notchFiltering(input_f, D0, u0, v0)

    [M, N] = size(input_f); % Get size of the image
    [U, V] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);

    % Compute distance matrices for notch filter centers
    D1 = sqrt((U - u0).^2 + (V - v0).^2);
    D2 = sqrt((U + u0).^2 + (V + v0).^2);

    % Create the notch filter (ideal filter)
    Notch = single((D1 >= D0) & (D2 >= D0));

    % Apply the notch filter in the frequency domain
    output_f = input_f .* Notch;
end
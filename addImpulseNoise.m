function output_s = addImpulseNoise(input_s, Ps, Pp)
    % Copy the input image
    output_s = input_s;
    [rows, cols] = size(input_s);
    
    % Generate random values for each pixel
    random_matrix = rand(rows, cols);
    
    % Set pixels to 255 (salt) for values less than Ps
    output_s(random_matrix < Ps) = 255;
    
    % Set pixels to 0 (pepper) for values between Ps and (Ps + Pp)
    output_s(random_matrix >= Ps & random_matrix < Ps + Pp) = 0;
end

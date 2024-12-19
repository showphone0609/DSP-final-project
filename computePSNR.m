function psnr = computePSNR(input1_s, input2_s)
    
    if ~isequal(size(input1_s), size(input2_s))
        error('Input images must have the same size and number of channels.');
    end

   
    input1_s = double(input1_s);
    input2_s = double(input2_s);

    %
    max_pixel = 255;

    
    psnr_channels = [];

    
    if size(input1_s, 3) == 3 
        for c = 1:3
            mse = mean((input1_s(:,:,c) - input2_s(:,:,c)).^2, 'all');
            if mse == 0
                psnr_channels(c) = Inf; 
            else
                psnr_channels(c) = 10 * log10(max_pixel^2 / mse); 
            end
        end
        
        psnr = mean(psnr_channels);
        fprintf('PSNR (R): %.2f dB, PSNR (G): %.2f dB, PSNR (B): %.2f dB\n', psnr_channels(1), psnr_channels(2), psnr_channels(3));
    else 
        mse = mean((input1_s - input2_s).^2, 'all');
        if mse == 0
            psnr = Inf; 
        else
            psnr = 10 * log10(max_pixel^2 / mse);
        end
    end
end

function convolve_output = convolve_input(filter_dim, filter_num, images, w, b)

    image_dim = size(images, 1);
    images_num = size(images, 4);
    conv_dim = image_dim - filter_dim + 1;
    
    convolve_output = zeros(image_dim, image_dim, filter_num, images_num);
    
    for i=1:images_num
        
            im_r = images(:, :, 1, i);
            im_g = images(:, :, 2, i);
            im_b = images(:, :, 3, i);
            
        for j=1:filter_num
            
            co = zeros(conv_dim, conv_dim);
            
            filter_r = w(:, :, 1, j);
            filter_g = w(:, :, 2, j);
            filter_b = w(:, :, 3, j);
            
            filter_r = rot90(filter_r, 2);
            filter_g = rot90(filter_g, 2);
            filter_b = rot90(filter_b, 2);
            
            co_r = conv2(imr_r, filter_r, 'valid');
            co_g = conv2(imr_g, filter_g, 'valid');
            co_b = conv2(imr_b, filter_b, 'valid');
            
            co = co_r + co_g + co_b;
            co = co + b(j);
            
            convolve_output(:, :, j, i) = relu(co);
        end
    end
end
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
function [gradients_x, gradients_y] = DetectImageGradients(img)

Gx = [1 0 -1; 2 0 -2; 1 0 -1];
Gy = [1 2 1; 0 0 0; -1 -2 -1];

gradients_x = conv2( double(img), double(Gx) );
gradients_y = conv2( double(img), double(Gy) );

% Convert to radians
gradients_x = gradients_x * pi / 180;
gradients_y = gradients_y * pi / 180;

end
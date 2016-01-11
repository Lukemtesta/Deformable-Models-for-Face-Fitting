function [samples, pixels] = MagnitudeAlongGradient(pt, width, gradient, values)

pt = floor(pt);
[x,y] = meshgrid(   pt(1)-width:width+pt(1),...
                    pt(2)-width:width+pt(2));
locations = [reshape(x, [], 1) reshape(y, [], 1)];
gradient = gradient ./ norm(gradient);
[best arr] = ClosestPointAlongVector(pt, locations, gradient);

for i=1:width*2 + 1
    [val, idx] = min(arr);
    pixels(i,:) = locations(idx,:);
    % Image is (y,x) and locations is (x,y)
    try
        samples(i) = values(pixels(i,2),pixels(i,1));
        arr(idx) = max(arr);
    catch
        % Outside image bounds - set to max value
        samples = ones(1, width*2 + 1) * bitmax;
        return;
    end
end

% Normalise array
samples = samples ./ norm(samples);

end
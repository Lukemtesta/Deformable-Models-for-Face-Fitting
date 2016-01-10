function keypoints = DetectFeatures(img, type, N, method)

if nargin < 3
    N = 200;
end

if nargin < 4
    method = 'MinimumEigenvalue';
end

if strcmp(type, 'HCD') == 1
   keypoints = corner(img, method, N);
end

if strcmp(type, 'CANNY') == 1
   [y, x] = find( edge(img) > 0 );
   keypoints = [x, y];
end

end
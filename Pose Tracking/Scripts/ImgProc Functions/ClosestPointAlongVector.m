function [closest_point dist_to_line] = ClosestPointAlongVector(pt, keypoints, gradient)

repeated_pts = repmat(pt, size(keypoints,1), size(keypoints,2)/2);
repeated_pts = repeated_pts - keypoints;
real_dist = sqrt( sum(repeated_pts .^ 2, 2) );

% Normalise longest components to max gradient
gradient = gradient ./ max( norm(gradient) );

% Reference : Wolfram Point-LineDistance2-Dimensional.html
for i=1:length(repeated_pts)
   repeated_pts(i,:) = repeated_pts(i,:) ./ max( norm(repeated_pts(i,:)) );
   numerator =  abs( gradient(1) * ( pt(2)-keypoints(i,2) ) - ...
                    (pt(1)-keypoints(i,1)) * gradient(2) );
   denominator = sqrt( sum( gradient .^ 2 ) );
   dist_to_line(i) = numerator / denominator;
end

% Find lowest error for equally weighted distance to pt and line metric
[val, idx] = min(dist_to_line);%' .* real_dist );
closest_point = keypoints(idx,:);

end
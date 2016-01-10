function error = FindLowestError(pt, keypoints)

repeated_pts = repmat(pt, size(keypoints,1), size(keypoints,2)/2);
repeated_pts = repeated_pts - keypoints;
error = min( sqrt( sum( (repeated_pts .^ 2)' ) ) ); 

end
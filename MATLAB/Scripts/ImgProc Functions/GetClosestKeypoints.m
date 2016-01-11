function [keypoints, idx] = GetClosestKeypoints(pt, candidates)

keypoints = repmat(pt, length(candidates), 1);
keypoints = keypoints - candidates;
keypoints = sqrt( sum( (keypoints .^ 2)' ) );
[val, idx] = min( keypoints );
keypoints = candidates(idx, :);

end

% Tried :   Closest Point, Closest Point Along mean gradient line in PCA
%           (lowest error so far),
%           Closest point along gradient of model on image pixels (worst
%           error with fewest iterations), 
%           Closest point along evolving gradient in PCA model (no error
%           and iteration improvement compared to mean gradient)
%           Closest point with common magnitude along gradient +/- in PCA
%           model
%
% To try:   Feature matching with another feature detector and descriptor
%           
%           


clear all, close all, clc
load ../Data/Annotations.mat

%% Load ground truths and image

TRAINING_PATH = '../Training Dataset/Annotated';
TRAINING_INDEX = 8;
KEYPOINTS_COUNT = 11;
EIGEN_WEIGHTING = 1.0;

for TRAINING_INDEX=1:8
    
    [ret, img] = LoadImage( [TRAINING_PATH '/' data.videos{TRAINING_INDEX}] );
    
    if ~ret
        error(  ['Could not load image ' ...
                pwd TRAINING_PATH '/' data.videos{TRAINING_INDEX}]);
    end
    
    [face_img, rects] = ViolaDetectFace(img);
    keypoints = DetectFeatures(face_img, 'CANNY');
    [gradients_x, gradients_y] = DetectImageGradients(face_img);
    G = sqrt(gradients_x .^ 2 + gradients_y .^ 2);
    %centroid = mean(keypoints);
    % commented out: Used for closest point along vector
    %mean_gradients = mean(data.gradients);
    
    % Build descriptor along each edge gradient for landmarks
    descriptors = [];
    samples = [];
    locations = [];
    for j=1:length(keypoints)
        pt = keypoints(j,:);
        % Invert keypoint x and y to get gradient in format (x,y)
        x = pt(2);
        y = pt(1);
        gradient = [gradients_x(x,y), gradients_y(x,y)];        
        % Take window around pixel and sample 5 mag of closest pixels on
        % gradient line
        [vals, locations] = MagnitudeAlongGradient(pt, EDGE_WINDOW_SIZE, gradient, G);
        gradient = gradient ./ norm(gradient);
        descriptors = [descriptors; gradient, vals];

    end
    
    %% Configure ASM
    ASM.iteration_min = 1e100;
    ASM.global_min = 1e99;
    sign = [-1,1];
    
    %% Map mean model to image space
    scale = length(face_img);
    %translation = mean(keypoints);
    
    x = E.U(1:KEYPOINTS_COUNT);
    y = E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT);

    mean_x = x * scale;% + repmat( translation(2), [1 length(x)] );
    mean_y = y * scale;% + repmat( translation(1), [1 length(y)] );
    
    % comment below = evolving PCA gradient for eigenvectors
    %gradient = E.U(2*KEYPOINTS_COUNT+1:length(E.vct));
    
    mean_gradient = mean(data.gradients);
    gradients = mean_gradient;
    x = mean_x;
    y = mean_y;
        
    %% Show mean model
    close all
    figure(1)
    imshow(face_img)
    hold on
    plot(mean_x, mean_y, 'r.', 'MarkerSize', 20);
    title('Mean Model')
    
    gt_err = [];
    closest_err = [];

    while ASM.iteration_min ~= ASM.global_min

        ASM.global_min = ASM.iteration_min
        
        candidates = []
        for i=1:length(x)
            idx = (i-1)*size(descriptors,2) + 1;
            test_gradients = gradients( idx:idx+size(descriptors,2)-1 );
            % commented below = gradient of scaled model pixels in image
            % idx = i*2 - 1;
            %gradient = [gradients_x(pt(2),pt(1)) gradients_y(pt(2),pt(1))]; 
%             candidates = [  candidates, ...
%                             ClosestPointAlongVector(pt, ...
%                                                     keypoints, ...                                             %mean_gradients(idx:idx+1))
%                                                     mean_gradients(idx:idx+1))
%                                                     %gradient(idx:idx+1))
%                                                     %gradient)
%                         ];
            test_gradients(:,3:end) = test_gradients(:,3:end) ./ ...
                                        norm(test_gradients(:,3:end));
            [pt, idx] =  GetClosestKeypoints(test_gradients(:,3:end), ...
                                                descriptors(:,3:end));
            candidates = [  candidates, keypoints(idx,:)];
        end

        % Brute force through modes to ensure PCA model was built correctly
        for v=1:length(E.vct)
            for s=1:length(sign)
                
                x_test =    x + (sign(s) * ...
                            E.vct(v, 1:KEYPOINTS_COUNT) * ...
                            EIGEN_WEIGHTING );
                        
                y_test =    y + (sign(s) * ...
                            E.vct(v, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * ...
                            EIGEN_WEIGHTING );
                        
                error = sqrt( sum( ( [x_test, y_test] - candidates) .^ 2 ) )
                
                if ASM.iteration_min > error
                    ASM.iteration_min = error;
                    ASM.min_sign = sign(s);
                    ASM.min_vector = v;
                end
            end
        end

        x = x + (   ASM.min_sign * ...
                    E.vct(ASM.min_vector, 1:KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
        y = y + (   ASM.min_sign * ...
                    E.vct(ASM.min_vector, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
        gradients = gradients + (   ASM.min_sign * ...
                    E.vct(ASM.min_vector, 23:end) * EIGEN_WEIGHTING );
        
        % comment below = evolving PCA gradient for eigenvectors
%         idx_start = 2*KEYPOINTS_COUNT+1;
%         idx_end = length(E.vct);
%         gradient = gradient + ...
%                     ( ASM.min_sign * ...
%                     E.vct(ASM.min_vector, idx_start:idx_end) * EIGEN_WEIGHTING );
                
        % Grab nearest neighbour and ground truth error data for regressor
%         diff_x = (x - mean_x) / length(face_img);
%         diff_y = (y - mean_y) / length(face_img);
%         euclidean = sqrt( sum( [ diff_x .^ 2, diff_y .^ 2 ] ) );
% 
%         gt_err = [gt_err; [ diff_x, ...
%                             diff_y, ...
%                             euclidean, ...
%                             error]];
% 
%         err = 0;
%         for j=1:length(x_test)
%            err = err + FindLowestError( ...
%                                 [x_test(j), y_test(j)], ...
%                                 keypoints );
%         end
% 
%         closest_err = [closest_err; [   diff_x, ...
%                                         diff_y, ...
%                                         euclidean, ...
%                                         err] ];
        
        figure(2)
        imshow(face_img)
        hold on
        plot(mean_x,mean_y,'r.','MarkerSize',20);
        mid = length(candidates)/2;
        plot(candidates(1:2:22), candidates(2:2:22), 'b.','MarkerSize',20)
        
%         figure(3)
%         imshow(face_img)
%         hold on
%         [x y] = find( edge(face_img) > 0 );
%         plot(y, x, 'g.', 'MarkerSize', 5);

    end
    
%     average_min_pos(:,:,TRAINING_INDEX) = gt_err(end,:);
% 
% 
%     %% Show ASM fit
% 
%     figure(3)
%     imshow(face_img)
%     hold on
%     plot([x], [y], 'r.', 'MarkerSize', 20);
%     title('ASM fit')
% 
%     % Write regressor data
%     csvwrite(['../Data/ground_truth_error_' num2str(TRAINING_INDEX) '.csv' ],...
%                 gt_err);
%     csvwrite(['../Data/nearest_neighbour_error_' num2str(TRAINING_INDEX) '.csv' ], ...
%                 closest_err);
        
end


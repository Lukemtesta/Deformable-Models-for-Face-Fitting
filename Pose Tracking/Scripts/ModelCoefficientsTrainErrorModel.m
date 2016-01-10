
clear all, close all, clc
load ../Data/Annotations.mat


%% Load ground truths and image

TRAINING_PATH = '../Training Dataset/Annotated';
TRAINING_INDEX = 8;
KEYPOINTS_COUNT = 19;

for TRAINING_INDEX=1:1
    
    [ret, img] = LoadImage( [TRAINING_PATH '/' data.videos{TRAINING_INDEX}] );
    [face_img, rects] = ViolaDetectFace(img);
    keypoints = DetectFeatures(face_img, 'CANNY');
    centroid = mean(keypoints);

    
    %% Configure ASM
    ASM.iteration_min = 1e100;
    ASM.global_min = 1e99;
    mean_gradients = mean(data.gradients);
    
    %% Map mean model to image space
    scale = length(face_img);
    translation = mean(keypoints);
    
    x = E.U(1:KEYPOINTS_COUNT);
    y = E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT);

    mean_x = x * scale + repmat( translation(2), [1 length(x)] );
    mean_y = y * scale + repmat( translation(1), [1 length(y)] );
    
    x = x * scale + repmat( translation(2), [1 length(x)] );
    y = y * scale + repmat( translation(1), [1 length(y)] );

    
    %% Show mean model

    figure(2)
    imshow(face_img)
    hold on
    plot(mean_x, mean_y, 'r.', 'MarkerSize', 20);
    title('Mean Model')
    
    gt_err = [];
    closest_err = [];

    while ASM.iteration_min ~= ASM.global_min

        ASM.global_min = ASM.iteration_min
        
        % Find closest point along model gradients       
        candidates = []
        
        for i=1:length(x)
            idx = i*2 - 1;
            pt = [x(i) y(i)];
            candidates = [  candidates, ...
                            ClosestPointAlongVector(pt, ...
                                                    keypoints, ...
                                                    mean_gradients(idx:idx+1))
                        ];
        end

        % Find eigenvalues for x = u + E.vct * b => b = E.vct' * (x - u)
        diff_x = candidates(1:2:length(candidates)) - mean_x;
        diff_y = candidates(2:2:length(candidates)) - mean_y;
        b = E.vct' * [ diff_x, diff_y ]';
        
        weighting = sum(E.vct .* repmat(b, [1, 38]));
        x = mean_x + weighting(1:19);
        y = mean_y + weighting(20:38);
        
        figure(3)
        imshow(face_img)
        plot(x,y,'r.','MarkerSize',20);

    end
    
    average_min_pos(:,:,TRAINING_INDEX) = gt_err(end,:);


    %% Show ASM fit

    figure(3)
    imshow(face_img)
    hold on
    plot([x], [y], 'r.', 'MarkerSize', 20);
    title('ASM fit')

    % Write regressor data
    csvwrite(['../Data/ground_truth_error_' num2str(TRAINING_INDEX) '.csv' ],...
                gt_err);
    csvwrite(['../Data/nearest_neighbour_error_' num2str(TRAINING_INDEX) '.csv' ], ...
                closest_err);
        
end


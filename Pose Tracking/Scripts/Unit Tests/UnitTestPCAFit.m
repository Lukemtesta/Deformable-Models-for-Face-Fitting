
clear all, close all, clc
load ../../Data/Annotations.mat

addpath('../Utility Functions')
addpath('../ImgProc Functions')

%% Load ground truths and image

TRAINING_PATH = '../../Training Dataset/Annotated';
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
    centroid = mean(keypoints);
    
    %% Configure ASM
    ASM.iteration_min = 1e100;
    ASM.global_min = 1e99;
    sign = [-1,1];
    
    %% Map mean model to image space
    scale = length(face_img);
    translation = mean(keypoints);
    
    x = E.U(1:KEYPOINTS_COUNT);
    y = E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT);

    mean_x = x * scale% + repmat( translation(2), [1 length(x)] );
    mean_y = y * scale% + repmat( translation(1), [1 length(y)] );
    
    x = mean_x;
    y = mean_y;
    
    candidates = data.ground_truths(TRAINING_INDEX, :);
    
    % Show ground truth
    figure(1)
    imshow(face_img);
    hold on
    mid = length(candidates)/2;
    plot(candidates(1:mid), candidates(mid+1:mid*2), 'r.','MarkerSize',20)
    
    %% Show mean model
    close all
    figure(2)
    imshow(face_img)
    hold on
    plot(mean_x, mean_y, 'r.', 'MarkerSize', 20);
    title('Mean Model')
    
    gt_err = [];
    closest_err = [];

    while ASM.iteration_min ~= ASM.global_min

        ASM.global_min = ASM.iteration_min

        % Brute force through modes to ensure PCA model was built correctly
        for v=1:length(E.vct)
            for s=1:length(sign)
                
                x_test =    x + (sign(s) * ...
                            E.vct(v, 1:KEYPOINTS_COUNT) * ...
                            EIGEN_WEIGHTING );
                        
                y_test =    y + (sign(s) * ...
                            E.vct(v, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * ...
                            EIGEN_WEIGHTING );
                        
                error = sqrt( sum( ( [x_test, y_test] - candidates) .^ 2 ) );
                
                if ASM.iteration_min > error
                    ASM.iteration_min = error
                    ASM.min_sign = sign(s);
                    ASM.min_vector = v;
                end
            end
        end

        x = x + (   ASM.min_sign * ...
                    E.vct(ASM.min_vector, 1:KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
        y = y + (   ASM.min_sign * ...
                    E.vct(ASM.min_vector, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * EIGEN_WEIGHTING );

        figure(3)
        imshow(face_img)
        hold on
        plot(x,y,'r.','MarkerSize',20);

    end
        
end


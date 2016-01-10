
clear all, close all, clc
load ../Data/Annotations.mat

addpath('Utility Functions')
addpath('ImgProc Functions')

%% Load ground truths and image

TRAINING_PATH = '../Training Dataset/Annotated';
TRAINING_INDEX = 8;
KEYPOINTS_COUNT = 11;
EIGEN_WEIGHTING = 1.0;

for TRAINING_INDEX=1:1
    
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
    translation = mean(keypoints);
    
    mean_x = E.U(1:KEYPOINTS_COUNT);
    mean_y = E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT);
    
    candidates = data.ground_truths(TRAINING_INDEX, :) / length(face_img);
    
    % Show ground truth
    close all
    figure(1)
    imshow(face_img);
    hold on
    mid = length(candidates)/2;
    plot(   candidates(1:mid) * length(face_img), ...
            candidates(mid+1:mid*2) * length(face_img), ... 
            'r.', ...
            'MarkerSize',20)
    title('Ground Truths')
    
    %% Show mean model
    figure(2)
    imshow(face_img)
    hold on
    plot(   mean_x * length(face_img), ...
            mean_y * length(face_img), ...
            'r.', ...
            'MarkerSize', 20);
    title('Mean Model')
    
    gt_err = [];
    closest_err = [];
    b = zeros(1, KEYPOINTS_COUNT * 2)';

    while ASM.iteration_min ~= ASM.global_min
        
        ASM.global_min = ASM.iteration_min
        
        % For modelling non-normalised mapping from image frame to PCA
        % Frame
        
        output =   [candidates(1:KEYPOINTS_COUNT); ...
                    candidates(1+KEYPOINTS_COUNT:KEYPOINTS_COUNT*2);
                    ones(1, KEYPOINTS_COUNT)]';
        input =    [mean_x; ...
                    mean_y; ...
                    ones(1, KEYPOINTS_COUNT)]';
                
        M = FindLowestErrorAffine(input, output);
        output = M' * output';
        
        %diff = [output(1,:) output(2,:)] - [mean_x mean_y];
        %diff = [output(1,:) output(2,:)] - [mean_x mean_y];

        % Below For modelling normalised mapping from image frame to PCA
        % Frame 
        diff = [candidates(1:KEYPOINTS_COUNT) ...
                candidates(1+KEYPOINTS_COUNT:KEYPOINTS_COUNT*2)] ...
                - [mean_x mean_y];
        b = E.vct' * diff';
        
        Pb = E.vct(1:22,1:22)' * b;
        Pb = Pb';
        
        x = mean_x + Pb(1:KEYPOINTS_COUNT);
        y = mean_y + Pb(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT);
        
        error = sqrt( sum( ( [x, y] - candidates) .^ 2 ) );
        
        if ASM.iteration_min > error
            ASM.iteration_min = error
        end
        
        figure(3)
        imshow(face_img)
        hold on
        plot(x * length(face_img),y * length(face_img),'r.','MarkerSize',20);

    end
        
end


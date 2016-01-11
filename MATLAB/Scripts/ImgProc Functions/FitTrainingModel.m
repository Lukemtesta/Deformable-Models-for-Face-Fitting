

clear all, close all, clc
load ../Data/Annotations.mat


%% Load ground truths and image

TRAINING_PATH = '../Training Dataset/Annotated';
TRAINING_INDEX = 8;
KEYPOINTS_COUNT = 15;
EIGEN_WEIGHTING = 0.5;

[ret, img] = LoadImage( [TRAINING_PATH '/' data.videos{TRAINING_INDEX}] );
img = imread('../Training Dataset/Annotated/12.jpg');

%img = imgaussfilt(img,2);
%img = histeq( rgb2gray(img) );

[face_img, rects] = ViolaDetectFace(img);
keypoints = DetectFeatures(face_img);
%[y x] = find( edge(face_img) > 0 );
%keypoints = [x, y];

%% Plot Detected Features

figure(1)
imshow(face_img)
hold on
plot(keypoints(:,1), keypoints(:,2), 'r.', 'MarkerSize', 20);
title('Keypoints')

%% Configure ASM
sign = [-1, 1];

ASM.iteration_min = 1e100;
ASM.global_min = 1e99;
ASM.min_sign = 0;
ASM.min_vector = 0;

%% Brute force ASM fitting

x = E.U(1:KEYPOINTS_COUNT) * length(face_img);
y = E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * length(face_img);
%x = start_pos(1:KEYPOINTS_COUNT) * length(face_img) + E.U(1:KEYPOINTS_COUNT) * length(face_img);
%y = start_pos(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * length(face_img) + E.U(1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * length(face_img);

%% Show mean model

figure(2)
imshow(face_img)
hold on
plot(x, y, 'r.', 'MarkerSize', 20);
title('Mean Model')

while ASM.iteration_min ~= ASM.global_min

    ASM.global_min = ASM.iteration_min
    
    for v=1:length(E.vct)
        for i=1:2

            x_test = x + ( sign(i) * E.vct(v, 1:KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
            y_test = y + ( sign(i) * E.vct(v, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * EIGEN_WEIGHTING );

            ground_truths = [];
            error = 0;
            for j=1:length(x_test)
               error = error + FindLowestError( ...
                                    [x_test(j), y_test(j)], ...
                                    keypoints ); 
            end
            
            error = error / length(x_test);

            if error < ASM.iteration_min
               ASM.iteration_min = error
               ASM.min_sign = i
               ASM.min_vector = v
            end
        end
    end
    
    x = x + (   sign(ASM.min_sign) * ...
                E.vct(ASM.min_vector, 1:KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
    y = y + (   sign(ASM.min_sign) * ...
                E.vct(ASM.min_vector, 1+KEYPOINTS_COUNT:2*KEYPOINTS_COUNT) * EIGEN_WEIGHTING );
        
end


%% Show ASM fit

figure(3)
imshow(face_img)
hold on
plot(x, y, 'r.', 'MarkerSize', 20);
title('ASM fit')


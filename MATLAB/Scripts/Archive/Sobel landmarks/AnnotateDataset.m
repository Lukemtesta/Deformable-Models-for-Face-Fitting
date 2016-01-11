
% Tried:        Translating by mean (poor starting position when fitting
%               to dataset due to mean of all edges), just scale invariant
%               (works best).
%
% To try:       Increase number of feature points along model contours!!

close all, clear all, clc

% Load images from the training dataset, find the face and 
% extract the normalised keypoint position relative to the top left
% corner of the face ROI
data.videos = [];
data.observations = [];
data.gradients = [];
data.ground_truths = [];

addpath('Utility Functions')
addpath('Unit Tests')
addpath('ImgProc Functions')

NUMBER_ANNOTATED_KEYPOINTS = 11;
TRAINING_IMAGE_COUNT = 11;
EDGE_WINDOW_SIZE = 2;
TRAINING_PATH = '../Training Dataset'

[ret, filenames] = GetImageFilenames(TRAINING_PATH);

for i=1:min(size(filenames,1),TRAINING_IMAGE_COUNT)

    % Load training images, detect a face extract keypoints
    [ret, img] = LoadImage( [TRAINING_PATH '\' filenames(i,:)] );
    [face_img, rects] = ViolaDetectFace(img);
    candidates = DetectFeatures(face_img, 'CANNY');
    [gradients_x, gradients_y] = DetectImageGradients(face_img);
    G = sqrt(gradients_x .^ 2 + gradients_y .^ 2);
    
    % Show image with detected keypoints
    close all
    imshow(face_img)
    ground_truths = ginput(NUMBER_ANNOTATED_KEYPOINTS);
    
    annotations = [];
    for j=1:NUMBER_ANNOTATED_KEYPOINTS
        annotations = [ annotations; ...
                            GetClosestKeypoints( ...
                                ground_truths(j,:), candidates)];
    end
    
    % Show image with annotated keypoints 
    hold on
    plot(annotations(:,1),annotations(:,2),'r.','MarkerSize',20) 
    
    gradients = [];
    samples = [];
    locations = [];
    for j=1:length(annotations)
        pt = annotations(j,:);
        % Invert keypoint x and y to get gradient in format (x,y)
        x = pt(2);
        y = pt(1);
        gradient = [gradients_x(x,y), gradients_y(x,y)];        
        % Take window around pixel and sample 5 mag of closest pixels on
        % gradient line
        [vals, locations] = MagnitudeAlongGradient(pt, EDGE_WINDOW_SIZE, gradient, G);
        gradient = gradient ./ norm(gradient);
        gradients = [gradients, gradient, vals];

    end
    
    % Normalise position based on face dimensions wrt world origin
    ground_truth = annotations;
    annotations(:,1) = annotations(:,1) / size(face_img,1);
    annotations(:,2) = annotations(:,2) / size(face_img,2);
    %annotations  =  annotations - ...
    %                repmat(mean(annotations), [length(annotations),1]);
    
    % If mouse click, store in database else throw data away
    k = waitforbuttonpress 
    if(k == 0)
        % store in row = features col = x <vidN>, y<vidN> ... 
        data.videos = [ data.videos; {filenames(i,:)} ];
        
        % Store un-modified ground truths
        data.ground_truths = [  data.ground_truths; ...
                                ground_truths(:,1)' ...
                                ground_truths(:,2)'];

        % re-arrange data for PCA where row = features (x -> xN,y -> yN), 
        % col = observation
        data.observations = [   data.observations; ...
                                annotations(:,1)', ...
                                annotations(:,2)'];
                            
        % store gradients as radians in pt1, pt2 ... ptN for with [Gx, Gy]
        data.gradients = [  data.gradients; ...
                            gradients];
    end
    
end

data.observations = [data.observations, data.gradients];
E = Eigen_Build(data.observations');
E.U = mean(data.observations);
save ../Data/Annotations.mat





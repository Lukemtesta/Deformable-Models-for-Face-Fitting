function [img, rects] = ViolaDetectFace(img)

faceDetector = vision.CascadeObjectDetector;
rects = step(faceDetector, img);

x = rects(1);
x_size = rects(3) + x;
y = rects(2);
y_size = rects(4) + y;

img = img(y:y_size, x:x_size);

end
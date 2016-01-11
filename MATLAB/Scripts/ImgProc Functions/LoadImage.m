function [ret, img] = LoadImage(img_path)

ret = false;
img = '';

if exist(img_path, 'file') ~= 2
   return 
end

img = imread(img_path);
ret = true;

end
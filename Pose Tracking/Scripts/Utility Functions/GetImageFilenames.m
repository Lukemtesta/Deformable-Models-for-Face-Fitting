
function [ ret, filenames ] = GetImageFilenames(filepath)

ret = false;
filenames = ls( [filepath '/*.jpg'] );

if(length(filenames) > 0)
   ret = true;
end

end
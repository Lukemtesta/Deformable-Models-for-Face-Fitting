function DisplayFace(img, rects)

x = rects(1);
x_size = rects(3) + x;
y = rects(2);
y_size = rects(4) + y;

imshow( img(y:y_size, x:x_size) );

end
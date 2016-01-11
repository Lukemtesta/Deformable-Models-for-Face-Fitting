function M = Derive2DAffineTransform(in, out)

if ( size(in,2) ~= 3 || size(out,2) ~= 3 )
    M = [];
    return;
end

X = [];
p = [];
% Limit aggine transform to avoid shear by describing rotation and scaling
% by a single vector : 
% http://stackoverflow.com/questions/11687281/transformation-between
% -two-set-of-points
for i=1:4
    X = [   X;
            in(i,1) in(i,2)    1    0;
            in(i,2) -in(i,1)   0    1];
    p = [p; out(i,1); out(i,2)];
end

a = pinv(X) * p;
M = [a(1) a(2) a(3); a(2) a(1) a(4); 0 0 1];

end
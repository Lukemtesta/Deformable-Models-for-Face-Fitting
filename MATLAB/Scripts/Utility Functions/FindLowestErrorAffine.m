function [ret, error] = FindLowestErrorAffine(in, out, range)

if nargin < 3
    range = 11;
end

min_error = 1e99;

for i=1:3300
    idx = randperm(11);
    idx = idx(1:4);
    M = Derive2DAffineTransform( in(idx,:), out(idx,:) ) ;
    error = ForwardBackwardError( in(idx,:), out(idx,:), M );
    
    if min_error > error
        min_error = error
        ret = M;
    end
end

end
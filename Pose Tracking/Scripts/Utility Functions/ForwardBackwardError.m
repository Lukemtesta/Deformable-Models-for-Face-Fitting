function error = ForwardBackwardError(in, out, M)

if ( size(in,2) ~= 3 || size(out,2) ~= 3 )
    error = [];
    return;
end

error = sum( ( abs(M * in')  + abs(M' * out') ) .^ 2 );
error = mean(error);

end
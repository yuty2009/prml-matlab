%% Compute the tangent line
function [xt,yt] = tangent(x0,y0,dy,width)
    a = dy;
    b = y0 - x0*a;
    xt = linspace(x0-width/2,x0+width/2,10);
    ff = @(x) a*x + b;
    yt = ff(xt);
end
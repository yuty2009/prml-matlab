function [X Y] = ellipse(a0, b0, a, b, angle, steps)
%# This functions returns points to draw an ellipse    
%#    
%#  @param a0    Center of X coordinate    
%#  @param b0    Center of Y coordinate    
%#  @param a     Semimajor axis    
%#  @param b     Semiminor axis    
%#  @param angle Rotation angle of the ellipse (in degrees)    
%#
%# Example:
%# p = ellipse(0, 0, 20, 10, 30);plot(p(:,1), p(:,2), '.-');axis equal;

error(nargchk(5, 6, nargin));
if nargin<6, steps = 36; end

beta = angle * (pi / 180);
sinbeta = sin(beta);
cosbeta = cos(beta);

alpha = linspace(0, 360, steps)' .* (pi / 180);
sinalpha = sin(alpha);
cosalpha = cos(alpha);

X = a0 + (a * cosalpha * cosbeta - b * sinalpha * sinbeta);
Y = b0 + (a * cosalpha * sinbeta + b * sinalpha * cosbeta);

if nargout==1, X = [X Y]; end

end
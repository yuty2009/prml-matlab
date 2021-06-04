function p = parzon(x, y, h1, fwin)

[N1, d] = size(x);
[N2, d_] = size(y);
assert(d == d_, 'x and y shape mismatch');

if ~exist('h1', 'var') h1 = 1; end
if ~exist('fwin', 'var') fwin = 'gauss'; end

h = h1 / sqrt(N1);
p = zeros(N2, 1);
for i = 1:N2
    p1 = 0;
    y1 = y(i,:);
    for j = 1:N1
        x1 = x(j,:);
        u = (y1 - x1) / h;
        if strcmp(fwin, 'gauss')
            % p1 = p1 + gauss(u);
            p1 = p1 + exp(-0.5*norm(u)^2)/sqrt(2*pi);
        elseif strcmp(fwin, 'cube')
            p1 = p1 + cube(u);
        else
            disp('unknown window type');
        end
        p(i) = p1 / (N1 * h^d);
    end
end

% function v = gauss(u)
%     v = exp(-0.5*norm(u)^2)/sqrt(2*pi);
% end
% 
% function v = cube(u)
%     ua = abs(u);
%     ub = ua <= 0.5;
%     ub = ub(:);
%     if prod(ub) > 0
%         v = 1;
%     else
%         v = 0;
%     end
% end
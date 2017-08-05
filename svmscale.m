function [X_scaled] = svmscale(X, bound, rangefile, option)
% X: N instances by P features
% bound: [-1 1] or [0 1]
% rangefile: filename used to save the current scale
% option:

% if dim(X)>=3, treat the dims except dim(1) as feature dim
dims = size(X);
X = X(:,:); 

if (option == 's') % scale
    
    smin = min(X);
    smax = max(X);
    
    index = 1:size(X,2);
    scale = [index; smin; smax];
    fp = fopen(rangefile, 'w');
    fprintf(fp, '%d %g %g\n', scale);
    fclose(fp);
    
elseif (option == 'r') % rescale with a range file
    
    load(rangefile, 'range');
    smin = range(:,2)';
    smax = range(:,3)';
    
end

rmin = min(bound);
rmax = max(bound);
sdiff = 1.0./(smax-smin+eps);

X_scaled = X;

for i = 1:size(X,1)
    X_scaled(i,:) = (rmax-rmin)*(X_scaled(i,:)-smin).*sdiff + rmin*ones(1,size(X,2));
    X_scaled(i,:) = (X_scaled(i,:) >= rmax)*rmax + (X_scaled(i,:) <= rmin)*rmin + ...
        ((X_scaled(i,:) > rmin) & (X_scaled(i,:) < rmax)).*X_scaled(i,:);
end
X_scaled = reshape(X_scaled,dims);

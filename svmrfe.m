%% SVM Recursive Feature Elimination
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NS: number of features/variables to be selected
%     FS: indices of selected features
%     References:
%     [1] I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, Gene selection
%     for cancer classification using support vector machines, Mach.
%     Learn., vol. 46, (no. 1-3), pp. 389-422, 2002.
function FS = svmrfe(y,X,NS)

[N,P] = size(X);

RFS = 1:P; % remain feature set
while length(RFS) > NS
    % model = svmtrain(y,X(:,RFS),'-s 0 -t 0 -c 1 -g 0.001');
    % w = model.SVs' * model.sv_coef;
    model = train(y,sparse(X(:,RFS)),'-s 2 -c 1');
    w = model.w;
    
    wsquare = w.^2;
    [dummy,index] = min(wsquare);
    RFS = setdiff(RFS,RFS(index));
end

FS = RFS;
%% Feature selection via SVM re-weighted iteration that approximates
%% L0-norm  SVM
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NS: number of features/variables to be selected
%     FS: indices of selected features
%     References:
%     [1] J. Weston, A.E. Elisseeff, B.S. Olkopf, and M. Tipping, Use of
%     the zero-norm with linear models and Kernel methods, J Mach Learn
%     Res, vol. 3, (no. 7-8), pp. 1439-1461, 2003.
function FS = svmznormf(y,X,NS)

[N,P] = size(X);

epsilon = 1e-9;
z = ones(P,1);
RFS = find(z>epsilon);
while length(RFS) > NS
    zMat = repmat(z,1,N);
    X1 = X.*zMat';
    model = svmtrain(y,X1,'-s 0 -t 0 -c 1 -g 0.001');
    w = model.SVs' * model.sv_coef;
    
    z = z.*w;
    RFS = find(z>epsilon);
end

FS = RFS;
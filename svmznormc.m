%% Channel selection via SVM re-weighted iteration that approximates
%% L0-norm  SVM
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NC: number of available channels
%     NS: number of channels to be selected
%     CS: indices of selected channels
%     FS: indices of selected features
%     References:
%     [1] J. Weston, A.E. Elisseeff, B.S. Olkopf, and M. Tipping, Use of
%     the zero-norm with linear models and Kernel methods, J Mach Learn
%     Res, vol. 3, (no. 7-8), pp. 1439-1461, 2003.
function [CS,FS] = svmznormc(y,X,NC,NS)

[N,P] = size(X);
P0 = P/NC;
IM = reshape(1:P,P0,NC); % indices in matrix form

% if the selected channel set does not channge in 5 iterations,
% terminate the iteration.
patience = 5; 
count = 0;
epsilon = 1e-9;
z = ones(P,1);
zAvg = mean(reshape(z,P0,NC));
RCS = find(zAvg>epsilon);
while (length(RCS) > NS) && (count < patience)
    zMat = repmat(z,1,N);
    X1 = X.*zMat';
    model = svmtrain(y,X1,'-s 0 -t 0 -c 1 -g 0.001');
    w = model.SVs' * model.sv_coef;
    
    z = z.*w; % update of the scaling factor
    zAvg = mean(reshape(z,P0,NC));
    RCSold = RCS;
    RCS = find(zAvg>epsilon);
    
    if isempty(setdiff(RCS,RCSold))
        count = count + 1;
    else
        count = 0;
    end
end

CS = RCS;
RFS = IM(:,CS);
FS = RFS(:);
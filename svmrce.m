%% SVM Recursive Channel Elimination
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NC: number of available channels
%     NS: number of channels to be selected
%     CS: indices of selected channels
%     FS: indices of selected features
%     References:
%     [1] T.N. Lal, M. Schroder, T. Hinterberger, J. Weston, M. Bogdan, N.
%     Birbaumer, and B. Scholkopf, Support vector channel selection
%     in BCI, IEEE Trans. Biomed. Eng., vol. 51, (no. 6), pp. 1003-10,
%     2004-06-01 2004.
function [CS,FS] = svmrce(y,X,NC,NS)

[N,P] = size(X);
P0 = P/NC;
IM = reshape(1:P,P0,NC); % indices in matrix form

RCS = 1:NC; % remain channel set
while length(RCS) > NS
    NR = length(RCS); %  number of remain channels
    RFS = IM(:,RCS);
    RFS = RFS(:); % remain feature set

    % model = svmtrain(y,X(:,RFS),'-s 0 -t 0 -c 1 -g 0.001');
    % w = model.SVs' * model.sv_coef;
    model = train(y,sparse(X(:,RFS)),'-s 2 -c 1');
    w = model.w;
    
    wsquare = w.^2;
    wsquareAvg = mean(reshape(wsquare,P0,NR));
    [dummy,index] = min(wsquareAvg);
    RCS = setdiff(RCS,RCS(index));
end

CS = RCS;
RFS = IM(:,CS);
FS = RFS(:);
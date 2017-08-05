function weight = searchlight(y,X,nbind,opts)

if nargin < 4
    opts.display = 0;
    opts.lambda = 1e-4;
end
if ~isfield(opts,'display'), opts.display = 0; end
if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end

[N,P] = size(X);

weight = zeros(P,1);
if opts.display == 1, hwait = waitbar(0,'Seachlight for feature 0'); end
for i = 1:P
    if opts.display == 1
        waitbar(i/P, hwait,['Seachlight for feature ' num2str(i) '/' num2str(P)]);
    end
    Xnb = X(:,nbind(:,i));
    % [w,b,weight(i)] = FLDA(y,Xnb,opts.lambda);
    [w,b,weight(i)] = RLDA(y,Xnb);
end
if opts.display == 1, close(hwait); end
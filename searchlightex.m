function weight = searchlightex(y,X,nbind,opts)

if nargin < 4
    opts.display = 0;
    opts.lambda = 1e-4;
end
if ~isfield(opts,'display'), opts.display = 0; end
if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end

P = size(X,3);

weight = zeros(P,1);
if opts.display == 1, hwait = waitbar(0,'Seachlight for feature 0'); end
for i = 1:P
    if opts.display == 1
        waitbar(i/P, hwait,['Seachlight for feature ' num2str(i) '/' num2str(P)]);
    end
    Xnb = X(:,:,nbind(:,i));
    Xnb = Xnb(:,:);
    
    index1 = find(y==1);
    index2 = find(y==-1);
    mX1 = mean(Xnb(index1,:));
    mX2 = mean(Xnb(index2,:));
    Sw = cov(Xnb);
    
    svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
    svmmodel = svmtrain(y,Xnb,svmoption);
    w = svmmodel.SVs' * svmmodel.sv_coef;
    pm1 = mX1*w;
	pm2 = mX2*w;
    weight(i) = (pm1-pm2)^2/(w'*Sw*w);
    
    % [w,b,weight(i)] = FLDA(y,Xnb,opts.lambda);
    % [w,b,weight(i)] = RLDA(y,Xnb);
end
if opts.display == 1, close(hwait); end
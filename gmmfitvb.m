%% Fitting the Gaussian mixture model give observations X
%% by variational Bayesian 
% input param
% X: X is a N-by-D matrix specifying N D-dimensional samples 
%    from Gaussian mixture model
% K: number of clusters

function gmmmodel = gmmfitvb(X,K,opts)

maxit = 500;
stopeps = 1e-6;
if nargin > 2
    if isfield(opts,'maxit')
        maxit = opts.maxit;
    end
    if isfield(opts,'stopeps')
        stopeps = opts.stopeps;
    end
else
    opts.verbose = 0;
    opts.plot = 0;
end

[N,D] = size(X);

% random initialization
% MU = X(1:K,:);
% SIGMA = reshape(repmat(eye(D),1,K),D,D,K);
% Pz = ones(1,K)/K;

% initialization with kmeans
% [IDX,MU] = kmeans(X,K);
% for k = 1:K
%     idx1 = find(IDX==k);
%     X1 = X(idx1,:);
%     SIGMA(:,:,k) = cov(X1);
%     Pz(k) = length(idx1)/N;
% end

% initialization with gmm EM
[MU,SIGMA,Pz] = gmmfit(X,K);

alpha0 = 1e-3;
beta0 = 1;
m0 = zeros(1,D);
W0 = 200*eye(D);
invW0 = W0^(-1);
v0 = 20;

NK = N*Pz;
alpha = alpha0 + NK;
beta = beta0 + NK;
v = v0 + NK;
for k = 1:K
    m(k,:) = (beta0*m0+NK(k)*MU(k,:))/beta(k);
    Sk = squeeze(SIGMA(:,:,k));
    W(:,:,k) = (invW0+NK(k)*Sk+beta0*NK(k)/(beta0+NK(k)) ...
        *(MU(k,:)-m0)'*(MU(k,:)-m0))^(-1);
end

Lold = -Inf;
for step = 1:maxit
    % E step
    E_mu_Lambda = zeros(N,K);
    E_ln_Lambda = zeros(K,1);
    E_ln_pi = zeros(K,1);
    for k = 1:K
        Wk = squeeze(W(:,:,k));
        for n = 1:N
            E_mu_Lambda(n,k) =  D/beta(k) + v(k)*(X(n,:)-MU(k,:))*Wk*(X(n,:)-MU(k,:))';
        end
        E_ln_Lambda(k) = sum(psi(0,0.5*(v(k)+1-[1:D]))) + D*log(2) + log(det(Wk));
        E_ln_pi(k) = psi(0,alpha(k)) - psi(0,sum(alpha));
    end
    item1 = E_ln_pi + 0.5*E_ln_Lambda; % - 0.5*D*log(2*pi);
    lnrho = repmat(item1',N,1) - 0.5*E_mu_Lambda;
    lnrho = bsxfun(@minus,lnrho,max(lnrho,[],2)); % to avoid underflow
    rho = exp(lnrho);
    r = bsxfun(@rdivide,rho,sum(rho,2));
    r = r + eps;

    % M step
    NK = sum(r);
    % MU = (r'*X)./repmat(NK',1,D);
    for k = 1:K
        MU(K,:) = r(:,k)'*X/NK(k);
        diff = X - repmat(MU(k,:),N,1);
        SIGMA(:,:,k) = diff'*diag(r(:,k))*diff/NK(k);
    end
    alpha = alpha0 + NK;
    beta = beta0 + NK;
    v = v0 + NK;
    for k = 1:K
        m(k,:) = (beta0*m0+NK(k)*MU(k,:))/beta(k);
        Sk = squeeze(SIGMA(:,:,k));
        W(:,:,k) = (invW0+NK(k)*Sk+beta0*NK(k)/(beta0+NK(k)) ...
            *(MU(k,:)-m0)'*(MU(k,:)-m0))^(-1);
    end
    
    % variational lower bound
    L1 = 0;
    L4 = 0;
    lnCalpha0 = gammaln(K*alpha0) - K*gammaln(alpha0);
    lnCalpha = gammaln(sum(alpha)) - sum(gammaln(alpha));
    for k = 1:K
        Wk = squeeze(W(:,:,k));
        Sk = squeeze(SIGMA(:,:,k)); 
        L1 = L1 + NK(k)*(E_ln_Lambda(k) - D/beta(k) - v(k)*trace(Sk*Wk) ...
            - v(k)*(MU(k,:)-m(k,:))*Wk*(MU(k,:)-m(k,:))' - D*log(2*pi));
        L4 = L4 + D*log(beta0/(2*pi)) + E_ln_Lambda(k) - D*beta0/beta(k) ...
            - beta0*v(k)*(m(k,:)-m0)*Wk*(m(k,:)-m0)' ...
            + (v0-D-1)*E_ln_Lambda(k) - v(k)*trace(W0^(-1)*Wk);
        lnBk = -(v(k)/2)*log(det(Wk)) - (v(k)*D/2)*log(2)...
            - (D*(D-1)/4)*log(pi) - sum(gammaln(0.5*(v(k)+1-[1:D])));
        H_q_Lambda(k) = -lnBk - 0.5*(v(k)-D-1)*E_ln_Lambda(k) + 0.5*v(k)*D;
    end
    lnB0 = -(v0/2)*log(det(W0)) - (v0*D/2)*log(2) ...
          - (D*(D-1)/4)*log(pi) - sum(gammaln(0.5*(v0+1-[1:D])));
    L1 = 0.5*L1;
    L2 = sum(r*E_ln_pi);
    L3 = lnCalpha0 + (alpha0-1)*sum(E_ln_pi);
    L4 = 0.5*L4 + K*lnB0;
    L5 = sum(sum(r.*log(r)));
    L6 = lnCalpha + sum((alpha-1).*E_ln_pi');
    L7 = sum(0.5*E_ln_Lambda'+0.5*D*log(beta)-0.5*D*log(2*pi)-0.5*D-H_q_Lambda);
    
    L(step) = L1+L2+L3+L4-L5-L6-L7;
    
    if isfield(opts,'verbose') && opts.verbose == 1
        disp(['Optimization step ' num2str(step), ...
            ', Variational lower bound = ' num2str(L(step))]);
    end
    
    if isfield(opts,'plot') && opts.plot == 1
        figure(3);clf;hold on;
        plot(X(:,1),X(:,2),'o');
        plot(m(:,1),m(:,2),'or','LineWidth',2);
        for k = 1:K
            error_ellipse(inv(W(:,:,k))/(v(k)-D-1), m(k,:));
            text(m(k,1), m(k,2), num2str(k),'BackgroundColor',[.7 .9 .7]);
        end
        pause(.01);
    end
    
    dL = abs(L(step)-Lold);
    Lold = L(step);
    if dL < stopeps
        break;
    end
end

gmmmodel.alpha = alpha;
gmmmodel.beta = beta;
gmmmodel.m = m;
gmmmodel.W = W;
gmmmodel.v = v;
gmmmodel.L = L;

%% Synthetic minority over-sampling technique
% Reference: N.V. Chawla, K.W. Bowyer, L.O. Hall, and W.P. Kegelmeyer, SMOTE:
% Synthetic minority over-sampling technique, J Artif Intell Res, vol. 16,
% pp. 321-357, 2002.
function Y = SMOTE(X,repeat,K,opts)

if nargin < 4
    opts.display = 0;
end
if ~isfield(opts,'display'), opts.display = 0; end

dims = size(X);
X = X(:,:);
[N,P] = size(X);

if opts.display == 1, hwait = waitbar(0,'Oversampling for sample 0'); end

Y = [];
for i = 1:N
    % find K nearest neighbors for each minority sample
    x1 = X(i,:);
    D1 = zeros(N-1,1);
    for j = setdiff(1:N,i)
        x2 = X(j,:);
        D1(j) = norm(x1-x2);
    end
    [dummy,idx] = sort(D1);
    nnbindex = idx(1:K);
    
    if opts.display == 1
        waitbar(i/N, hwait,['Oversampling for sample ' num2str(i) '/' num2str(N)]);
    end
    
    y1 = synthetic(x1, repeat, X(nnbindex,:));
    Y = cat(1,Y,y1);
end
if opts.display == 1, close(hwait); end
Y = reshape(Y,[size(Y,1),dims(2:end)]);

end

function y1 = synthetic(x1,repeat,Xnb)
    [K,P] = size(Xnb);
    y1 = zeros(repeat,P);
    for i = 1:repeat
        k = ceil(K*rand(1)+eps);
        for j = 1:P
            gamma = rand(1);
            y1(i,j) = (1-gamma)*x1(j)+gamma*Xnb(k,j);
        end
    end
end

%% Calculate the TPR and FPR for a Receiver Operating Characteristic analysis
%  y: N by 1 ground truth labels (-1,1)
%  yp: N by 1 predicted scores (real-values)
%  TPR and FPR are calculated with set each of the sorted scores as 
%  Example: [TPR,FPR,AUC] = ROC(y,yp);
function varargout = ROC(y,scores)

N = length(scores);
N1 = length(find(y==1));
N2 = length(find(y==-1));

[scores,idx]=sort(scores);
theta = scores;
y = y(idx);

E1 = 0;
E2 = N2;
FP = zeros(N,1);
TP = zeros(N,1);
for i = 1:N
  if y(i) == 1
    E1 = E1 + 1;
  else
    E2 = E2 - 1;
  end
  
  FP(i) = E2/N2;
  TP(i) = (N1-E1)/N1;
end

sigma = 0;
for i = N1+N2:-1:1
    if y(i) == 1
        sigma = sigma + i;
    end
end
AUC = (sigma-(N1+1)*N1/2)/(N1*N2);

varargout{1} = TP;
varargout{2} = FP;
if nargout > 2
    varargout{3} = AUC;
elseif nargout > 3
    varargout{4} = theta;
end

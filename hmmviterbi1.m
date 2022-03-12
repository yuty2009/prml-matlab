function [Z] = hmmviterbi1(X,prior,transmat,emissmat)
%#   hmmrnd calculates the most probable state path for a sequence.
%#   [Z] = hmmviterbi(X,prior,transmat,emissmat) given a sequence, X,
%#   calculates the most likely path through the Hidden Markov Model
%#   specified by transition probability matrix, prior and transmat, 
%#   and EMISSION probability matrix, emissmat. 
%#   transmat(I,J) is the probability of transition from state
%#   I to state J. obsmat(K,D) is the probability that value D is
%#   emitted from state K. 
%#   example:

L = length(X);
[K D] = size(emissmat);

log_prior = log(prior);
log_transmat = log(transmat);
log_emissmat = log(emissmat);

Zforward = zeros(L,K);

Zforward(1,:) = zeros(1,K);
w(1,:) = log_prior + log_emissmat(:,X(1))';

for i = 2:L
    w1 = zeros(1,K);
    for j = 1:K
        [w1(j) Zforward(i,j)] = max(log_transmat(:,j)'+w(i-1,:));
    end
    w(i,:) = log_emissmat(:,X(i))' + w1;
end

[dummy Zfinal] = max(w(L,:));

Z(L) = Zfinal;
for i = L-1:-1:1
    Z(i) = Zforward(i+1,Z(i+1));
end
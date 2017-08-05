function [X Z] = hmmrnd(prior,transmat,emissmat,L)
%#   hmmrnd generate a sequence for a Hidden Markov Model
%#   [X Z] = hmmrnd(prior,transmat,emissmat,L) generates
%#   sequence of emission, X, and a random sequence of states,
%#   Z, of length L from a Markov Model specified by transition
%#   probability matrix, prior and transmat, and EMISSION probability matrix,
%#   emissmat. transmat(I,J) is the probability of transition from state
%#   I to state J. obsmat(K,D) is the probability that value D is
%#   emitted from state K. 
%#   example:
%      prior = [0.5 0.5];
%      transmat = [0.95,0.05;
%                  0.10,0.90];
%           
% 	   emissmat = [1/6,  1/6,  1/6,  1/6,  1/6,  1/6;
%                  1/10, 1/10, 1/10, 1/10, 1/10, 1/2;];
%      [X Z] = hmmrnd(prior,transmat,emissmat,100);

[K,D] = size(emissmat);

Z(1) = randptable(prior);
X(1) = randptable(emissmat(Z(1),:));

for i = 2:L
    Zlast = Z(i-1);
    Z(i) = randptable(transmat(Zlast,:));
    X(i) = randptable(emissmat(Z(i),:));
end
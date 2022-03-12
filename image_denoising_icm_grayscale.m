%% Image denoising with Iterated Conditional Modes from PRML markov random
%% field [Page.389].
% $ \[E(x) = \alpha {\rm{ }}\sum\limits_{{x_j} \in N({x_i})} {\left\| {{x_i} - {x_j}} \right\|_2^2} 
% + {\rm{ }}\beta \sum\limits_i {\left\| {{x_i} - {y_i}} \right\|_2^2} \] $

clc
clear

im = imread('image_denoising_noisy_1.bmp');
Y = double(im);
X = Y;

alpha = 100.0;
beta = 1.0;
[M N] = size(Y);
for iterator = 1:1
    for i = 1:M
        for j = 1:N  
            [Nb len] = neighbors(X,i,j);
            X(i,j) = (alpha*sum(Nb)+beta*Y(i,j))/(alpha*len+beta);
        end
    end
end
imnew = uint8(X);
imbwnew = im2bw(imnew);

subplot(131);imshow(im);
subplot(132);imshow(imnew);
subplot(133);imshow(imbwnew);
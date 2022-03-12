%% Image denoising with Iterated Conditional Modes from PRML markov random
%% field [Page.389].
% the energe function defined below should be minimized
% $ \[E(x) = h\sum\limits_i {{x_i}}  - \beta \sum\limits_{{x_j} \in N({x_i})} {{x_i}{x_j}}  - \eta \sum\limits_i {{x_i}{y_i}} \] $

clc
clear

im = imread('image_denoising_noisy_1.bmp');
imbw = im2bw(im);
for i = 1:size(imbw,1)
        for j = 1:size(imbw,2)
            if imbw(i,j) == 1
                Y(i,j) = 1;
            else
                Y(i,j) = -1;
            end
        end
end
X = Y;

h = 0;
beta = 1.0;
nta = 2.0;
[M N] = size(Y);
for iterator = 1:1
    for i = 1:M
        for j = 1:N
            X_n = -1;
            X_p =  1;   
            [E0 E1] = neighbour_energy(X,i,j);
            E_n = h*X_n - beta*E0 - nta*X_n*Y(i,j);
            E_p = h*X_p - beta*E1 - nta*X_p*Y(i,j);
            if E_n < E_p
                X(i,j) = X_n;
            else
                X(i,j) = X_p;
            end
        end
    end
end
imbw_denoisy = im2bw(X);

subplot(121);imshow(imbw);
subplot(122);imshow(imbw_denoisy);
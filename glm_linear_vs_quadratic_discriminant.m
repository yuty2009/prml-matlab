%% Figure 4.11 Page 200 PRML
clc
clear

%% Generate samples
mu1 = [1 -1];
mu2 = [-1 -1];
mu3 = [0 1.5];
SIGMA1 = [0.5 0; 0 1];
SIGMA2 = [0.5 0; 0 1];
SIGMA3 = [1 0; 0 0.5];%[0.5 0; 0 1];%

x1 = [-2.5:0.1:2.5]';
x2 = [-2.5:0.1:2.5]';
M = length(x1);
N = length(x2);
Pxgc1 = zeros(M,N);
Pxgc2 = zeros(M,N);
Pxgc3 = zeros(M,N);
for i = 1:M
    for j = 1:N
        Pxgc1(i,j) = mvnpdf([x1(i),x2(j)],mu1,SIGMA1);
        Pxgc2(i,j) = mvnpdf([x1(i),x2(j)],mu2,SIGMA2);
        Pxgc3(i,j) = mvnpdf([x1(i),x2(j)],mu3,SIGMA3);
    end
end

%% According to equation 4.68-4.70 Page 199
% assume P(C1) = P(C2) = P(C3) = 1/3

% w1 = SIGMA1^(-1)*mu1';
% w2 = SIGMA2^(-1)*mu2';
% w3 = SIGMA3^(-1)*mu3';
% w10 = -1/2*mu1*SIGMA1^(-1)*mu1' + log(1/3);
% w20 = -1/2*mu2*SIGMA2^(-1)*mu2' + log(1/3);
% w30 = -1/2*mu3*SIGMA3^(-1)*mu3' + log(1/3);
% Pcgx1 = zeros(M,N);
% Pcgx2 = zeros(M,N);
% Pcgx3 = zeros(M,N);
% C = zeros(M,N,3);
% for i = 1:M
%     for j = 1:N
%         x = [x1(i) x2(j)];
%         alpha1 = w1'*x' + w10;
%         alpha2 = w2'*x' + w20;
%         alpha3 = w3'*x' + w30;
%         Pcgx1(i,j) = exp(alpha1)/(exp(alpha1)+exp(alpha2)+exp(alpha3));
%         Pcgx2(i,j) = exp(alpha2)/(exp(alpha1)+exp(alpha2)+exp(alpha3));
%         Pcgx3(i,j) = exp(alpha3)/(exp(alpha1)+exp(alpha2)+exp(alpha3));
%         C(i,j,:) = [Pcgx1(i,j) Pcgx2(i,j) Pcgx3(i,j)];
%     end
% end

Pc1 = 1/3;
Pc2 = 1/3;
Pc3 = 1/3;
Pcgx1 = zeros(M,N);
Pcgx2 = zeros(M,N);
Pcgx3 = zeros(M,N);
C = zeros(M,N,3);
for i = 1:M
    for j = 1:N
        x = [x1(i) x2(j)];
        Pcgx1(i,j) = Pxgc1(i,j)*Pc1/(Pxgc1(i,j)*Pc1+Pxgc2(i,j)*Pc2+Pxgc3(i,j)*Pc3);
        Pcgx2(i,j) = Pxgc2(i,j)*Pc2/(Pxgc1(i,j)*Pc1+Pxgc2(i,j)*Pc2+Pxgc3(i,j)*Pc3);
        Pcgx3(i,j) = Pxgc3(i,j)*Pc3/(Pxgc1(i,j)*Pc1+Pxgc2(i,j)*Pc2+Pxgc3(i,j)*Pc3);
        C(i,j,:) = [Pcgx1(i,j) Pcgx2(i,j) Pcgx3(i,j)];
    end
end

%% Visualization
figure;
subplot(121);
hold on; box on;
contour(x1,x2,Pxgc1,3);
contour(x1,x2,Pxgc2,3);
contour(x1,x2,Pxgc3,3);
subplot(122);
surf(x1,x2,Pxgc1,C,'EdgeColor','none');
xlim([-2.5 2.5]);view(2);
%% Figure 4.10 Page 199 PRML
clc
clear

%% Generate samples
mu1 = [-0.5 -0.5];
mu2 = [0.5 0.5];
SIGMA1 = [0.5 0; 0 0.5];
SIGMA2 = [0.5 0; 0 0.5];

x1 = [-1:0.1:1]';
x2 = [-1:0.1:1]';
M = length(x1);
N = length(x2);
Pxgc1 = zeros(M,N);
Pxgc2 = zeros(M,N);
C1 = zeros(M,N,3);
C2 = zeros(M,N,3);
for i = 1:M
    for j = 1:N
        Pxgc1(i,j) = mvnpdf([x1(i),x2(j)],mu1,SIGMA1);
        Pxgc2(i,j) = mvnpdf([x1(i),x2(j)],mu2,SIGMA2);
        C1(i,j,:) = [1 0.5-Pxgc1(i,j) 0.5-Pxgc1(i,j)];
        C2(i,j,:) = [0.5-Pxgc2(i,j) 0.5-Pxgc2(i,j) 1];
    end
end

%% According to equation 4.65-4.67 Page 198
% assume P(C1) = P(C2) = 1/2

% w = SIGMA1^(-1)*(mu1-mu2)';
% w0 = -1/2*mu1*SIGMA1^(-1)*mu1' + 1/2*mu2*SIGMA1^(-1)*mu2';
% Pcgx1 = zeros(M,N);
% C3 = zeros(M,N,3);
% for i = 1:M
%     for j = 1:N
%         x = [x1(i) x2(j)];
%         alpha = w'*x' + w0;
%         Pcgx1(i,j) = 1/(1+exp(-alpha));
%         C3(i,j,:) = [Pcgx1(i,j) 0 1-Pcgx1(i,j)];
%     end
% end

Pc1 = 1/2;
Pc2 = 1/2;
Pcgx1 = zeros(M,N);
C3 = zeros(M,N,3);
for i = 1:M
    for j = 1:N
        Pcgx1(i,j) = Pxgc1(i,j)*Pc1/(Pxgc1(i,j)*Pc1+Pxgc2(i,j)*Pc2);
        C3(i,j,:) = [Pcgx1(i,j) 0 1-Pcgx1(i,j)];
    end
end

%% Viusalization
figure;
subplot(121);
hold on;grid on;
surf(x1, x2, Pxgc1, C1,'EdgeColor','none');
surf(x1, x2, Pxgc2, C2,'EdgeColor','none');
view(45,15);
subplot(122);
surf(x1, x2, Pcgx1, C3,'EdgeColor','none');
view(45,15);
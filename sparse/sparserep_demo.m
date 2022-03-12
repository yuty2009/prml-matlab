clc
clear

load('elec35_nor');
x = signals(32,:)';
k = 35;

% representation using a dictionary consisting of the Daubechies extremal
% phase wavelet and scaling vectors at level 2, the discrete cosine
% transform basis, a sine basis, the Kronecker delta basis, and the
% Daubechies least asymmetric phase wavelet and scaling vectors with 4
% vanishing moments at levels 1 and 4.
dict1 = {{'db4',2},'dct','sin',{'sym4',1},{'sym4',4}};
[mpdict1,nbvect1] = wmpdictionary(length(x),'lstcpt',dict1);
% y1 = wmpalg('OMP',x,mpdict1,'itermax',k);
[w1,y1] = OMP(x,mpdict1,k);

% representation using DCT-sine dictionary 
dict2 = {'dct','sin'};
[mpdict2,nbvect2] = wmpdictionary(length(x),'lstcpt',dict2);
% y2 = wmpalg('OMP',x,mpdict2,'itermax',k);
[w2,y2] = OMP(x,mpdict2,k);

% representation using DFT basis
xdft = fft(x);
[dummy,I] = sort(xdft(1:length(x)/2+1),'descend');
idx = I(1:k);
indconj = length(xdft)-idx+2;
idx = [idx indconj];
xdftapp = zeros(size(xdft));
xdftapp(idx) = xdft(idx);
y3 = ifft(xdftapp);

figure;
subplot(311);
hold on;
plot(x);
plot(y1,'r');
xlabel('Minutes'); ylabel('Usage');
legend('raw signal','OMP','Location','NorthEast');
set(gca,'xlim',[1 1440]);

subplot(312);
hold on;
plot(x);
plot(y2,'r');
xlabel('Minutes'); ylabel('Usage');
legend('raw signal','OMP using DCT-sine basis','Location','NorthEast');
set(gca,'xlim',[1 1440]);

subplot(313);
hold on;
plot(x);
plot(y3,'r');
xlabel('Minutes'); ylabel('Usage');
legend('raw signal','OMP using DFT basis','Location','NorthEast');
set(gca,'xlim',[1 1440]);
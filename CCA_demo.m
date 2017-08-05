clc
clear

load carbig;
X = [Displacement Horsepower Weight Acceleration MPG];
nans = sum(isnan(X),2) > 0;
[A1 B1 r1 U1 V1] = canoncorr(X(~nans,1:3), X(~nans,4:5));
[A2 B2 r2 U2 V2] = CCA(X(~nans,1:3), X(~nans,4:5));

figure;
subplot(121);
plot(U1(:,1),V1(:,1),'.');
xlabel('0.0025*Disp + 0.020*HP - 0.000025*Wgt');
ylabel('-0.17*Accel + -0.092*MPG')

subplot(122);
plot(U2(:,1),V2(:,1),'.');
xlabel('0.0025*Disp + 0.020*HP - 0.000025*Wgt');
ylabel('-0.17*Accel + -0.092*MPG')
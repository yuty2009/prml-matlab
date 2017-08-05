clc
clear

N1 = 50;
N2 = 500;

r1 = sqrt(abs(randn(N1,1))); % Radius
a1 = 2*pi*rand(N1,1);  % Angle
X1 = [r1.*cos(a1), r1.*sin(a1)]; % Points
y1 = ones(N1,1);

r2 = sqrt(abs(3*randn(N2,1))+2); % Radius
a2 = 2*pi*rand(N2,1);      % Angle
X2 = [r2.*cos(a2), r2.*sin(a2)]; % points
y2 = -ones(N2,1);

repeat = (N2-N1)/N1;
X11 = SMOTE(X1,repeat,6);
y11 = ones(N1*repeat,1);

figure;
hold on
plot(X1(:,1),X1(:,2),'r.','MarkerSize',10)
plot(X2(:,1),X2(:,2),'b.','MarkerSize',10)
plot(X11(:,1),X11(:,2),'g.','MarkerSize',10)
axis([-5 5 -5 5]);
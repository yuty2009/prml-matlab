%% Page 350, Fig 7.10 of PRML
clc
clear

PHI = [-2 4]';
t = [4 2]';

beta = 5;
alpha = 50;

a1 = -5:0.1:5;
a2 = -5:0.1:5;

Ibeta = (1/beta)*eye(size(PHI,1));
C = Ibeta + (1/alpha)*PHI*PHI';

d0 = zeros(length(a1),length(a2));
d1 = zeros(length(a1),length(a2));
for i = 1:length(a1)
    for j = 1:length(a2)
        d0(i,j) = [a1(i) a2(j)]*Ibeta^(-1)*[a1(i) a2(j)]';
        d1(i,j) = [a1(i) a2(j)]*C^(-1)*[a1(i) a2(j)]';
    end
end

figure;
hold on;grid on;
scatter(t(1),t(2),'x','LineWidth',2);
contour(a1,a2,d0,10.1,'g--');
contour(a1,a2,d1,10.1,'r-');
plot([0 PHI(1)],[0 PHI(2)],'k');
axis([-5 5 -5 5]);
axis square;

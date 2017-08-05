clc
clear

N1 = 10;
N2 = 10;
x1 = 8 + randi(12,N1,1);
x2 = 8 - randi(8,N2,1);

% [h1,p1] = permtest(x1,8,0.05,'right');
% [h2,p2] = permtest(x2,8,0.05,'left');
[h1,p1] = ttest(x1,8,0.05,'right');
[h2,p2] = ttest(x2,8,0.05,'left');

figure;
boxplot([x1 x2]);
line([0 2.5],[8 8]);
text(1.2,16,['mean1 > 8, p = ' num2str(p1)]);
text(1.0,5,['mean2 < 8, p = ' num2str(p2)]);
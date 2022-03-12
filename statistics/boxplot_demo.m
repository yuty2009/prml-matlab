
N1 = 12;
N2 = 8;

X1 = randi(8,[N1,1]) + 10;
X2 = 10 - randi(8,[N2,1]);

N = max([N1,N2]);
X = NaN(N,2);
X(1:N1,1) = X1;
X(1:N2,2) = X2;

[h,p,ci,stat] = ttest2(X1,X2,0.05,'right');

figure;
subplot(131);
hold on;
boxplot(X);
ylim([0,20]);
text(1.2,19,['T(' num2str(stat.df) ') = ' num2str(stat.tstat) ', p = ' num2str(p)]);
subplot(132);
hold on;
boxplot(X1);
ylim([0,20]);
subplot(133);
hold on;
boxplot(X2);
ylim([0,20]);
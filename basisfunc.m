%% basis function visualization Page 295, Fig. 6.1 of PRML

clc
clear

const = 0;
order = 1:10;
mu = -1:0.2:1;
sigma = 0.2;
a = 10;
b = -1:0.2:1;

x1 = -1:0.01:1;
x2 = -1:0.01:1;
index0 = find(x1==0);
index05 = find(x1==-0.5);

for i = 1:length(order)
    bf1(:,i) = (x1+const).^order(i);
end
for i = 1:length(x1)
    for j = 1:length(x1)
        K1(i,j) = bf1(i,:)*bf1(j,:)';
    end
end
    
for i = 1:length(mu)
    bf2(:,i) = exp(-(x1-mu(i)).*(x1-mu(i))/(2*sigma^2));
end
for i = 1:length(x1)
    for j = 1:length(x1)
        K2(i,j) = bf2(i,:)*bf2(j,:)';
    end
end

for i = 1:length(b)
    bf3(:,i) = 1./(1+exp(-a*(x1+b(i))));
end
for i = 1:length(x1)
    for j = 1:length(x1)
        K3(i,j) = bf3(i,:)*bf3(j,:)';
    end
end

figure;
subplot(331);
hold on;
for i = 1:length(order)
    plot(x1,bf1(:,i),'Color',[(10-order(i))/10 0 order(i)/10]);
end
subplot(332);
hold on;
for i = 1:length(mu)
    plot(x1,bf2(:,i),'Color',[(1-mu(i))/2 0 (mu(i)+1)/2]);
end
subplot(333);
hold on;
for i = 1:length(b)
    plot(x1,bf3(:,i),'Color',[(1-b(i))/2 0 (b(i)+1)/2]);
end

subplot(334);
mesh(K1);
axis tight; view(2);
subplot(335);
mesh(K2);
axis tight; view(2);
subplot(336);
mesh(K3);
axis tight; view(2);

subplot(337);
plot(x1,K1(index05,:));
subplot(338);
plot(x1,K2(index0,:));
subplot(339);
plot(x1,K3(index0,:));
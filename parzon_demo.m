clc;
clear;

% x0 = randn(100000, 1);
% y = -3:0.01:3; y = y';
% Ns = [10, 100, 1000];
% hs = [0.2, 0.5, 1];
% figure;
% for i = 1:length(Ns)
%     N = Ns(i);
%     x = x0(1:N, :);
%     for j = 1:length(hs)
%         h = hs(j);
%         p = parzon(x, y, h);
%         subplot(length(Ns), length(hs), (i-1)*length(hs)+j);
%         plot(y, p, 'k-');
%         title(['n = ' num2str(N) ', h = ' num2str(h)]);
%     end
% end

x0 = mvnrnd([0, 0], eye(2), 100000);
y1 = -3:0.25:3;
y2 = -3:0.25:3;
[y1, y2] = meshgrid(y1', y2');
y = [y1(:), y2(:)];
Ns = [10, 100, 5000];
hs = [0.2, 1, 5];
figure;
for i = 1:length(Ns)
    N = Ns(i);
    x = x0(1:N, :);
    for j = 1:length(hs)
        h = hs(j);
        p = parzon(x, y, h);
        subplot(length(Ns), length(hs), (i-1)*length(hs)+j);
        surf(y1, y2, reshape(p, size(y1)));
        title(['n = ' num2str(N) ', h = ' num2str(h)]);
    end
end
        
clc
clear

fx = @(x) 0.5*x.^2;
dfx = @(x) x; % gradient

x = -10:0.1:10;
y = fx(x);

x0 = -8; % initial value
eta = 0.1; % step
maxit = 50; % number of iterations
x1 = x0;

figure;
for i = 1:maxit
    y1 = fx(x1);
    dy1 = dfx(x1);
    [xt,yt] = tangent(x1,y1,dy1,3);

    subplot(121);
    hold off;
    plot(x,y,'k');
    hold on;
    scatter(x1,y1,50,'o','MarkerEdgeColor','r');
    plot(xt,yt,'b');
    ylim([-5 50]);
    xlabel('x');
    ylabel('y = 0.5*x^2');
    title('optimization path');
    
    gx(i) = i;
    gy(i) = dy1;
    subplot(122);
    plot(gx,gy);
    axis([1 maxit -10 0]);
    xlabel('iteration');
    title('gradient');
    
    pause(1);

    x1 = x1 - eta*dy1; % update x
end
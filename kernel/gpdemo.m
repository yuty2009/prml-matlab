clc
clear

gpargs = {
    [1,4,0,0];
    [9,4,0,0];
    [1,64,0,0];
    [1,0.25,0,0];
    [1,4,10,0];
    [1,4,0,5]
};

X = linspace(-1,1,100)';

for i = 1:length(gpargs)
    opts.args = gpargs{i};
    subplot(2,3,i);hold on;
    for j = 1:4
        y = gpsample(X,opts);
        plot(X,y);
    end
    title(num2str(gpargs{i}));
end
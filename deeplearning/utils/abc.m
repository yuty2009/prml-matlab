clc
clear

% img = rand(100, 28, 28);
% lbl = rand(100,10);
% 
% ds = imageset(img, lbl);
% for i = 1:15
%     [bx, by] = ds.nextbatch(20, true);
% end

datapath = 'e:\prmldata\mnist\';
mnist = mnistdata();
[train, validation, test] = mnist.load(datapath, true, false);

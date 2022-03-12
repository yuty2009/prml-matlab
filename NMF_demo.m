clc
clear

im = imread('lena.jpg');
imdouble = double(im);

rs = [4 16 64 128 256];
imrecon = cell(length(rs),1);
for i = 1:length(rs)
    disp(['rank ' num2str(rs(i))]);
    [W, H] = NMF(imdouble, rs(i), 500);
    imtemp = W*H;
    imrecon{i} = uint8(imtemp);
end

%% visualization
subplot(2,3,1);
imshow(im);
title('raw');
for i = 1:length(rs)
    subplot(2,3,i+1);
    imshow(imrecon{i});
    title(['rank = ' num2str(rs(i))]);
end
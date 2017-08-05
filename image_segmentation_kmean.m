clc
clear

im = imread('image_segmentation_raw_1.bmp');
X = zeros(size(im,1)*size(im,2), size(im,3));
for i = 1:size(im,1)
    for j = 1:size(im,2)
        X((i-1)*size(im,2)+j,:) = im(i,j,:);
    end
end

k = 10;
[idx,ctrs] = kmeans(X, k);
ctrs = uint8(ctrs);
for i = 1:size(im,1)
    for j = 1:size(im,2)
        imseg(i,j,:) = ctrs(idx((i-1)*size(im,2)+j), :);
    end
end

subplot(121);imshow(im);
subplot(122);imshow(imseg);
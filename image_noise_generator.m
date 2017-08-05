

percent = 0.1;
im = imread('image_denoising_raw.bmp');
imbw = im2bw(im);

for i=1:size(imbw,1)
    for j=1:size(imbw,2)
        dummy = randint(1,1,[0 99]);
        if dummy < 100*percent
            imbw_noisy(i,j) = 1 - imbw(i,j);
        else
            imbw_noisy(i,j) = imbw(i,j);
        end
    end
end

imwrite(imbw_noisy, 'image_denoising_noisy_1.bmp', 'bmp');

subplot(121);imshow(imbw);
subplot(122);imshow(imbw_noisy);

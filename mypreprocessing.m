%% 1-read original Image
I = dicomread('G:\BreastCancerWorking\Datasets\1-1.dcm');
figure
imshow(I,[]);
%%
I= x20586908;
%%
% [L,N] = superpixels(I,500);
% %Display the superpixel boundaries overlaid on the original image.
% 
% figure
% BW = boundarymask(L);
% imshow(imoverlay(I,BW,'cyan'),'InitialMagnification',67)
% %%
% outputImage = zeros(size(I),'like',I);
% idx = label2idx(L);
% numRows = size(I,1);
% numCols = size(I,2);
% for labelVal = 1:N
%     redIdx = idx{labelVal};
%     greenIdx = idx{labelVal}+numRows*numCols;
%     blueIdx = idx{labelVal}+2*numRows*numCols;
%     outputImage(redIdx) = mean(I(redIdx));
%     outputImage(greenIdx) = mean(I(greenIdx));
%     outputImage(blueIdx) = mean(I(blueIdx));
% end    
% 
% figure
% imshow(outputImage,'InitialMagnification',67)
%% 2-apply gussian filter to get D_not
% D_not=imgaussfilt(I,'FilterSize',[3 3]);
% imshow(D_not)
% %% 3- Subtract D_not from Original Image to get L_not
% L_not = I-D_not;
% imshow(L_not)
% %% 4- Downsample to D_not by two and produce L_1 by similarly applying guassian filters 
% % and subtractions, repeat it to get D_1,2,3,4, and L_1,2,3,4
% [r c m]=size(I);
% r_1=round(r/2);
% c_1=round(c/2);
% downsample_D_one=imresize(I,[r_1 c_1]);


%% To create Laplacian images, we need to downsample the original image as well 
[r c m]=size(I);
r_1=round(r/2);
c_1=round(c/2);
downsample_I_one=imresize(I,[r_1 c_1]);
[r c m]=size(downsample_I_one);
r_2=round(r/2);
c_2=round(c/2);
downsample_I_two=imresize(downsample_I_one,[r_2 c_2]);
[r c m]=size(downsample_I_two);
r_3=round(r/2);
c_3=round(c/2);
downsample_I_three=imresize(downsample_I_two,[r_3 c_3]);
[r c m]=size(downsample_I_three);
r_4=round(r/2);
c_4=round(c/2);
downsample_I_four=imresize(downsample_I_two,[r_4 c_4]);
[r c m]=size(downsample_I_four);
r_5=round(r/2);
c_5=round(c/2);
residual_Image=imresize(downsample_I_four,[r_5 c_5]);
% %% Create D's by doing gussian blur on downsamples images
% D_one=imgaussfilt(downsample_I_one,'FilterSize',[7 7]);
% D_two=imgaussfilt(downsample_I_two,'FilterSize',[5 5]);
% D_three=imgaussfilt(downsample_I_three,'FilterSize',[5 5]);
% D_four=imgaussfilt(downsample_I_four,'FilterSize',[3 3]);
% %% here is downsampled creation of L_1,2,3 and 4
% L_one = adapthisteq(downsample_I_one);
% L_two = adapthisteq(downsample_I_two);
% L_three = adapthisteq(downsample_I_three);
% L_four = adapthisteq(downsample_I_four);
residual_Image = adapthisteq(residual_Image);

%% Medium frequency components are in L1, L2 and L3
% Apply mean on laplacian images
% mx_not = mean(D_not);
% mx_one = mean(D_one);
% mx_two = mean(D_two);
% mx_three = mean(D_three);
% mx_four = mean(D_four);
% % %% get global standard deviations of Laplacian images
% std_not=std2(double(mx_not));
% std_one=std2(double(mx_one));
% std_two=std2(double(mx_two));
% std_three=std2(double(mx_three));
% std_four=std2(double(mx_four));
% %% get subtraction of laplacian images from mx,1,2,3,4 and divide them by their standard deviations
% temp_not=double(L_not)-mx_not;
% temp_one=double(L_one)-mx_one;
% temp_two=double(L_two)-mx_two;
% temp_three=double(L_three)-mx_three;
% temp_four=double(L_four)-mx_four;
% %% division
% div_not=temp_not/std_not;
% div_one=temp_one/std_one;
% div_two=temp_two/std_two;
% div_three=temp_three/std_three;
% div_four=temp_four/std_four;
% %% K is the modified contrast gain factor
% % There are certain conditions for K-factor
% % case-1: if K < std then contrast is reduced
% % case-2: if K = std then contrast will remain same
% % case-3: if K > std then contrast is enhanced
% %% As we need to enhance the given image. Therefore, will apply std based greater value
% % Let K should be the 1.5x to the calculated std of each Laplacian Image
% K_not=std_not.*4/2+std_not;
% K_one=std_one.*5/2+std_one;
% K_two=std_two.*5/2+std_two;
% K_three=std_three*5/2+std_three;
% K_four=std_four;
% %% apply histeq on laplacian images
% 
% hist_not = adapthisteq(I);
% hist_one = adapthisteq(downsample_I_one);
% hist_two = adapthisteq(downsample_I_two);
% hist_three = adapthisteq(downsample_I_three);
% hist_four = adapthisteq(downsample_I_four);
% %% Not calculate the L' for each Laplacian Image
% LL_not=double(mx_not)+K_not.*temp_not;
% LL_one=double(mx_one)+K_one.*temp_one;
% LL_two=double(mx_two)+K_two.*temp_two;
% LL_three=double(mx_three)+K_three.*temp_three;
% LL_four=double(mx_four)+K_four.*temp_four;
%% Upsampling now
[r c m]=size(downsample_I_four);
residual_upsample=imresize(residual_Image,[r c],'bicubic');
% residual_filtered=imgaussfilt(residual_upsample,'FilterSize',[3 3]);
% residual_upsample=residual_upsample+L_four;
%% Obtain Reconsutructed Laplacian Images
% R_four=residual_filtered;
%%
[r_3 c_3 m]=size(downsample_I_three);
upsample_three=imresize(residual_upsample,[r_3 c_3],'bicubic');
% L_3_filtered=imgaussfilt(upsample_three,'FilterSize',[3 3]);
%  R_three=upsample_three+uint16(L_three);
%%
[r_2 c_2 m]=size(downsample_I_two);
upsample_two=imresize(upsample_three,[r_2 c_2],'bicubic');
% L_2_filtered=imgaussfilt(upsample_two,'FilterSize',[3 3]);
%R_two=upsample_two.*uint16(L_two);
%%
[r_1 c_1 m]=size(downsample_I_one);
upsample_one=imresize(upsample_two,[r_1 c_1],'bicubic');
% L_one_filtered=imgaussfilt(upsample_one,'FilterSize',[3 3]);
 %R_one=upsample_one.*uint16(L_one);
%%
[r_0 c_0 m]=size(I);
upsample_not=imresize(upsample_one,[r_0 c_0],'bicubic');
% L_not_filtered=imgaussfilt(upsample_not,'FilterSize',[3 3]);
%R_not=upsample_not;
%% now showing original and reconstructed Image
imshowpair(I,histeq(I-(upsample_not-I)),'montage')
title('Original Vs Laplacian-SDMACE using 3x3 Gussian')

%%
% J = adapthisteq(I);
% 
% imshowpair(I,J,'montage');
%title('Original Image (left) and Contrast Enhanced Image (right)')
%%
final=upsample_not-I;
imshowpair(I,adapthisteq(imsharpen(imsharpen(final+I))),'montage');
%%
figure,
subplot(231)
imshow(I)
title('Original Image')
subplot(232)
imshow(adapthisteq(I))
title('CLAHE Image')
subplot(233)
imshow(imsharpen(adapthisteq(final+I)))
title('Proposed Enhanced Image')
subplot(234)
imhist(I)
title('Original Image Histogram')
subplot(235)
imhist(adapthisteq(I))
title('CLAHE Enhanced Image Histogram')
subplot(236)
imhist(imsharpen(adapthisteq(final+I)))
title('Proposed Image Histogram')
%%
[peaksnr, snr] = psnr(cont, I);

fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The SNR value is %0.4f', snr);
%%
smooth=imgaussfilt(I,'FilterSize',[5 5]);
sharpening=imsharpen(imsharpen(smooth));
contrast_Adj=imadjust(sharpening,[0.2 0.7],[]);
cont=adapthisteq(contrast_Adj);


figure,
subplot(231)
imshow(I)
title('Original Image')
subplot(232)
imshow(smooth)
title('smooth Image')
subplot(233)
imshow(sharpening)
title('sharpening')
subplot(234)
imshow(contrast_Adj)
title('Contrast')
subplot(235)
imshow(cont)
title('CLAHE')
subplot(236)
imshow(imgaussfilt(cont,'FilterSize',[7 7]))
title('Proposed Image Histogram')
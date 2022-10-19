%% 1-read original Image
I = dicomread('G:\BreastCancerWorking\Datasets\1-1.dcm');
figure
imshow(I,[]);

%% 2-apply gussian filter to get D_not
D_not=imgaussfilt(I,'FilterSize',[3 3]);
imshow(D_not)
%% 3- Subtract D_not from Original Image to get L_not
L_not = I-D_not;
imshow(L_not)
%% 4- Downsample to D_not by two and produce L_1 by similarly applying guassian filters 
% and subtractions, repeat it to get D_1,2,3,4, and L_1,2,3,4
[r c m]=size(I);
r_1=round(r/2);
c_1=round(c/2);
downsample_D_one=imresize(D_not,[r_1 c_1]);


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
%% Create D's by doing gussian blur on downsamples images
D_one=imgaussfilt(downsample_I_one,'FilterSize',[3 3]);
D_two=imgaussfilt(downsample_I_two,'FilterSize',[3 3]);
D_three=imgaussfilt(downsample_I_three,'FilterSize',[3 3]);
D_four=imgaussfilt(downsample_I_four,'FilterSize',[3 3]);
%% here is downsampled creation of L_1,2,3 and 4
L_one = downsample_I_one-D_one;
L_two = downsample_I_two-D_two;
L_three = downsample_I_three-D_three;
L_four = downsample_I_four-D_four;
%% Medium frequency components are in L1, L2 and L3
%% Apply Alpha trimmed mean filter on laplacian images
mx_not = getAlphatrimmed(L_not);
mx_one = getAlphatrimmed(L_one);
mx_two = getAlphatrimmed(L_two);
mx_three = getAlphatrimmed(L_three);
mx_four = getAlphatrimmed(L_four);
%% get global standard deviations of Laplacian images
std_not=std2(double(L_not));
std_one=std2(double(L_one));
std_two=std2(double(L_two));
std_three=std2(double(L_three));
std_four=std2(double(L_four));
%% get subtraction of laplacian images from mx,1,2,3,4 and divide them by their standard deviations
temp_not=double(L_not)-mx_not;
temp_one=double(L_one)-mx_one;
temp_two=double(L_two)-mx_two;
temp_three=double(L_three)-mx_three;
temp_four=double(L_four)-mx_four;
%% division
div_not=temp_not/std_not;
div_one=temp_one/std_one;
div_two=temp_two/std_two;
div_three=temp_three/std_three;
div_four=temp_four/std_four;
%% K is the modified contrast gain factor
% There are certain conditions for K-factor
% case-1: if K < std then contrast is reduced
% case-2: if K = std then contrast will remain same
% case-3: if K > std then contrast is enhanced
%% As we need to enhance the given image. Therefore, will apply std based greater value
% Let K should be the 1.5x to the calculated std of each Laplacian Image
K_not=std_not.*4/2+std_not;
K_one=std_one.*4/2+std_one;
K_two=std_two.*5/2+std_two;
K_three=std_three*5/2+std_three;
K_four=std_four;
%% Not calculate the L' for each Laplacian Image
LL_not=mx_not+K_not.*temp_not;
LL_one=mx_one+K_one.*temp_one;
LL_two=mx_two+K_two.*temp_two;
LL_three=mx_three+K_three.*temp_three;
LL_four=mx_four+K_four.*temp_four;
%% Upsampling now
[r c m]=size(downsample_I_four);
residual_upsample=imresize(residual_Image,[r c],'bicubic');
residual_filtered=imgaussfilt(residual_upsample,'FilterSize',[3 3]);
%% Obtain Reconsutructed Laplacian Images
R_four=residual_filtered+uint16(LL_four);
%%
[r_3 c_3 m]=size(downsample_I_three);
upsample_three=imresize(R_four,[r_3 c_3],'bicubic');
L_3_filtered=imgaussfilt(upsample_three,'FilterSize',[3 3]);
R_three=L_3_filtered+uint16(LL_three);
%%
[r_2 c_2 m]=size(downsample_I_two);
upsample_two=imresize(R_three,[r_2 c_2],'bicubic');
L_2_filtered=imgaussfilt(upsample_two,'FilterSize',[3 3]);
R_two=L_2_filtered+uint16(LL_two);
%%
[r_1 c_1 m]=size(downsample_I_one);
upsample_one=imresize(R_two,[r_1 c_1],'bicubic');
L_one_filtered=imgaussfilt(upsample_one,'FilterSize',[3 3]);
R_one=L_one_filtered+uint16(LL_one);
%%
[r_0 c_0 m]=size(I);
upsample_not=imresize(R_two,[r_0 c_0],'bicubic');
L_not_filtered=imgaussfilt(upsample_not,'FilterSize',[3 3]);
R_not=L_not_filtered+uint16(LL_not);
%% now showing original and reconstructed Image
imshowpair(I,R_not-I,'montage')
title('Original Vs Laplacian-SDMACE using 3x3 Gussian')

%%
J = adapthisteq(I);


imshowpair(I,J,'montage');
title('Original Image (left) and Contrast Enhanced Image (right)')
%%
final=R_not-I;
imshowpair(I,final,'montage');
%%
figure,
subplot(221)
imshow(I)
title('Original Image')
subplot(222)
imshow(J+I)
title('Restored Image')
subplot(223)
imhist(I)
title('Original Image histogram')
subplot(224)
imhist(J+I)
title('Restored Image Histogram')
d = dir('G:\BreastCancerWorking\Datasets\INbreast\Classes_Data\Malignant\*.jpeg'); % 
%d1 = dir('G:\INbreast Release 1.0\classesData\masks\Benign\*.jpg');
% d1 = dir('G:\INbreast Release 1.0\post-proccesed-writing\Benign+Malignant Masks\Benign- Masses-Predicted Segmentation\*.jpg'); % 

for ii=1:length(d) 
    imgfiles=strcat('G:\BreastCancerWorking\Datasets\INbreast\Classes_Data\Malignant\',d(ii).name); 
    %maskFiles = strcat('G:\INbreast Release 1.0\post-proccesed-writing\Benign+Malignant Masks\Benign- Masses-Predicted Segmentation\',d1(ii).name);
    img=imread(imgfiles);
%     mask=imread(maskFiles);
     I=imresize(img,[512 512]);
     
     [r_2 c_2 m]=size(I)
     if(m>1)
         I=rgb2gray(I);
     else
         I=I;
     end
%     bw= imbinarize(I);
%      thresh=img2;
%     bw(bw==1)=255;
% %     mask2= imresize(mask,[512 512]);
    %C = semanticseg(img2, net);
    %bw=C=='tumor';
    %bw2=bwareafilt(bw,[2000 Inf]);
 %    bw=(imbinarize(img2));
%     BW = imfill(mask2,'holes');
%     se = strel('disk',10);
%     dilatedBW = logical(imdilate(BW,se));
%     new= uint8(dilatedBW).*(img2);
%    thresh=uint8(thresh);

%     smooth=imgaussfilt(I,'FilterSize',[5 5]);
%     sharpening=imsharpen(imsharpen(smooth));
%     contrast_Adj=imadjust(sharpening,[0.2 0.7],[]);
%     cont=adapthisteq(contrast_Adj);
%     smooth2=imgaussfilt(cont,'FilterSize',[7 7]);
    %newStr = erase( d(ii).name , '.bmp' );
    smooth=imgaussfilt(I,'FilterSize',[5 5]);
    sharpening=imsharpen(imsharpen(smooth));
    contrast_Adj=imadjust(sharpening,[0.2 0.7],[]);
    cont=adapthisteq(contrast_Adj);
    
    
    newfilename=strcat('G:\BreastCancerWorking\Datasets\INbreast\Classes_Data\Malignant\',d(ii).name);
    newstr=erase(newfilename,'.jpeg');
    imwrite(cont,[newstr,'.jpg']);
end
%%
img=imread('51049053.jpg');

BW = imfill(img,'holes');
se = strel('disk',10);
dilatedBW = imdilate(BW,se);
figure,imshow(dilatedBW)
d = dir('G:\python_for_microscopists-master\Inbreast\patches_256\masks\*.jpg'); % 
%d1 = dir('G:\INbreast Release 1.0\classesData\masks\Benign\*.jpg');
% d1 = dir('G:\INbreast Release 1.0\post-proccesed-writing\Benign+Malignant Masks\Benign- Masses-Predicted Segmentation\*.jpg'); % 

for ii=1:length(d) 
    imgfiles=strcat('G:\python_for_microscopists-master\Inbreast\patches_256\masks\',d(ii).name); 
    %maskFiles = strcat('G:\INbreast Release 1.0\post-proccesed-writing\Benign+Malignant Masks\Benign- Masses-Predicted Segmentation\',d1(ii).name);
    img=imread(imgfiles);
    bw=imbinarize(img);
%     mask=imread(maskFiles);
    % img2=imresize(img,[512 512]);
    if (length(unique(img))>1)
     bw=imbinarize(img);
   % bw(bw==1)=255;
        %thresh=uint8(bw);
    end
%     mask2= imresize(mask,[512 512]);
    %C = semanticseg(img2, net);
    %bw=C=='tumor';
    %bw2=bwareafilt(bw,[2000 Inf]);
 %    bw=(imbinarize(img2));
%     BW = imfill(mask2,'holes');
%     se = strel('disk',10);
%     dilatedBW = logical(imdilate(BW,se));
%     new= uint8(dilatedBW).*(img2);
%     thresh=img;
     newStr = erase( d(ii).name , '.jpg' );
    newfilename=strcat('G:\python_for_microscopists-master\Inbreast\m_patches_256\masks\',d(ii).name);
    newStr = erase( newfilename , '.jpg' );
    imwrite(bw,[newStr '.png']);
end
%%
img=imread('51049053.jpg');

BW = imfill(img,'holes');
se = strel('disk',10);
dilatedBW = imdilate(BW,se);
figure,imshow(dilatedBW)
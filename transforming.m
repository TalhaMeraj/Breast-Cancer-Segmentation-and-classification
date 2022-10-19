  %  clear all
    img=imread('21_LEFT_MLO.png');%reading image.
    %img=rgb2gray(img);
    imshow(img);
%%
    level=graythresh(img);
    x=im2bw(img,level);
    figure,imshow(x);
%%
    measurements = regionprops(im2double(x), 'Centroid')
    figure(), imshow(x,'border','tight'), hold on,
    plot(measurements(1).Centroid(1),measurements(1).Centroid(2),'or','MarkerFaceColor','r')
 %%
    [rows columns] = size(x);
    %%
    rowsToShift = round(rows/2- measurements.Centroid(2));
    columnsToShift = round(columns/2 - measurements.Centroid(1));
 %%
    shiftedImage = circshift(img, [-1 -1]);
    figure,
    imshow(shiftedImage, []);
 %%
    nnew=img.*cast(x,class(img));
    medimage=medfilt2(nnew,[5 5]);%median Filtering image
    figure,imshow(medimage);
    z=adapthisteq(medimage);
    figure,imshow(z);
    H = fspecial('unsharp');
    y=imfilter(z,H);
    figure,imshow(y);
    %%
    mIm=imfilter(y,fspecial('average',31),'replicate');

    sIm=y-mIm;
    bw=im2bw(sIm,0);                % Convert to binary image
    ALT_img=imcomplement(bw);          % Complement binary image
    figure,imshow(ALT_img);
    %%
    %%%%%%morphological on binary images.

    open_img =bwmorph(ALT_img,'open'  ,Inf);
    figure,imshow(open_img);title('open_img');
    major_img =bwmorph(open_img,'majority' ,Inf);
    figure,imshow(major_img);title('major_img');
    ske_img = bwmorph(major_img,'skel',100);
    figure,imshow(ske_img);
    %%
    % Find the left-most binary pixel
[rows, columns] = find(x);  % Coordinates of all the white pixels.
% Extract just the left most one.
xLeft = (size(columns)+1)/2;
yLeft = (size(rows)+1)/2;
% Plot it
hold on;
plot(xLeft, yLeft, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
%%
FindMiddle(x) 
plot(xLeft, yLeft, 'r*', 'MarkerSize', 20, 'LineWidth', 3);

%%
function middleElement = FindMiddle(squareArray) 
% FindMiddle: Return the element in the center of squareArray
%   Inputs: squareArray - n x n input array, where n is odd
%
%   Outputs: selectedData - center element of squareArray
    
    % Assign elementIndex with location of middle row/col
    % Hint: Use the size() function to deterimine the dimension of squareArray
   center_ind = (size(squareArray)+1)/2;
    
    % Assign middleElement with the center element of squareArray
    middleElement = squareArray(center_ind(1), center_ind(2))
end
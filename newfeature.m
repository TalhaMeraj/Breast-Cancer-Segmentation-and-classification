function [all_feature]=newfeature(img)
img=imresize(img,[512 512]);
[cA,cH1,cV1,cD1] = dwt2(img,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
%sizeinmm=double(sum(img(:))*0.264583333);
% % Inverse Difference Movement
% % Inverse Difference Movement
% m = size(G,1);
% n = size(G,2);
% in_diff = 0;
% for i = 1:m
%     for j = 1:n
%         temp = G(i,j)./(1+(i-j).^2);
%         in_diff = in_diff+temp;
%     end
% end
%HOGfeatures = double(extractHOGFeatures(img));
%SURFfeatures = detectSURFFeatures(img);
%IDM = double(in_diff);  
all_feature = [Kurtosis,Skewness,Contrast,Correlation,Energy,Homogeneity,Mean,Standard_Deviation,Entropy,RMS,Smoothness,Variance];


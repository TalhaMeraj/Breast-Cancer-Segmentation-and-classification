function HOGfeatures=hog_extractor(Image)
img=imresize(Image,[512 512]);
[cA,cH1,cV1,cD1] = dwt2(img,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
DWT_feat = [cA3,cH3,cV3,cD3];
HOGfeatures = double(extractHOGFeatures(DWT_feat));

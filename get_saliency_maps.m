

img = x1878_RIGHT_MLO_Copy;
figure;
subplot(1, 4, 1);
imagesc(img);
title('Original Image');
axis off;
          
[Saliency_Map, Feature_Maps, ICA_Maps, img] = Run_SUN(img, []);
          
subplot(1, 4, 2);imagesc(img);title('Preprocessed Image');
          
axis off;
         
subplot(1, 4, 3);
imagesc(Saliency_Map);
title('Saliency Map');
subplot(1, 4, 4);
imagesc(mean(Feature_Maps, 3));
axis off;
title('Mean Feature Map');
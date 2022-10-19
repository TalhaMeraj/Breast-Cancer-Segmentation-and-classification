imageFolder = fullfile('G:\BreastCancerWorking\Datasets\CBIS-DDSM\Class-wise-data\predicted\','All');

imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%%
[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
%%
for i=2767:length(imds.Labels)
    I=readimage(imds,i);
    [Saliency_Map, Feature_Maps, ICA_Maps, img, SUN] = Run_SUN(I);
    saliency_features{i}=mean(Feature_Maps, 3);
end
%%
for i=1:length(saliency_features)
    saliency_features_mean(i,:)=mean(saliency_features{i});
end
%%

YLabel=imds.Labels;
rng default    % Set the seed for reproducibility
c = cvpartition(YLabel,'HoldOut',0.2);
idxTrain = training(c,1);
idxTest = ~idxTrain;
XTrain = saliency_features_mean(idxTrain,:);
yTrain = YLabel(idxTrain);
XTest = saliency_features_mean(idxTest,:);
yTest = YLabel(idxTest);
%%
saliency_mat=[saliency_features_mean,grp2idx(imds.Labels)];

%%
 svmmL=fitcecoc(XTrain,yTrain);

YPred = predict(svmmL,XTest);
accuracy = mean(YPred == yTest)
%%


 logitt=fitcensemble(saliency_features_mean,imds.Labels,'Method','LogitBoost','KFold',10);
ce = kfoldLoss(logitt);
accuracy=1-ce

%%
svmguasian1 = fitcsvm(saliency_features_mean,imds.Labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto','KFold',10);
ce = kfoldLoss(svmguasian1);
accuracy=1-ce
%%
knnfine = fitcknn(saliency_features_mean,imds.Labels,'NumNeighbors',1,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knnfine);
accuracy=1-ce
%%
knnmedium = fitcknn(saliency_features_mean,imds.Labels,'NumNeighbors',10,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knnmedium);
accuracy=1-ce
%%
knncoarse = fitcknn(saliency_features_mean,imds.Labels,'NumNeighbors',100,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knncoarse);
accuracy=1-ce
%%
bagmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','Bag','KFold',10);
ce = kfoldLoss(bagmodel);
accuracy=1-ce
%%
totalboostmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','TotalBoost','KFold',10);
ce = kfoldLoss(totalboostmodel);
accuracy=1-ce
%%
gentleboostmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','GentleBoost','KFold',10);
ce = kfoldLoss(gentleboostmodel);
accuracy=1-ce
%%
robustboostmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','RobustBoost','KFold',10);
ce = kfoldLoss(robustboostmodel);
accuracy=1-ce
%%
LPboostmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','LPBoost','KFold',10);
ce = kfoldLoss(LPboostmodel);
accuracy=1-ce
%%
RUSboostmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','RUSBoost','KFold',10);
ce = kfoldLoss(RUSboostmodel);
accuracy=1-ce
%%
SubSpacemodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','Subspace','KFold',10);
ce = kfoldLoss(SubSpacemodel);
accuracy=1-ce

%%
denNet=densenet201();
%%
imageSize = denNet.Layers(1).InputSize;
augmentedSet = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');
%%
w1 = denNet.Layers(3).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)

%%
featureLayer = 'fc1000';
deepFeatures = activations(denNet, augmentedSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
%%
saliency_dense_features=[deepFeatures,saliency_features_mean];
%%
logitt=fitcensemble(saliency_dense_features,imds.Labels,'Method','LogitBoost','KFold',10);
ce = kfoldLoss(logitt);
accuracy=1-ce
%%
svmguasian1 = fitcsvm(saliency_dense_features,imds.Labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto','KFold',10);
ce = kfoldLoss(svmguasian1);
accuracy=1-ce
%%
knnfine = fitcknn(saliency_dense_features,imds.Labels,'NumNeighbors',1,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knnfine);
accuracy=1-ce
%%
knnmedium = fitcknn(saliency_dense_features,imds.Labels,'NumNeighbors',10,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knnmedium);
accuracy=1-ce
%%
knncoarse = fitcknn(saliency_dense_features,imds.Labels,'NumNeighbors',100,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
ce = kfoldLoss(knncoarse);
accuracy=1-ce
%%
bagmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','Bag','KFold',10);
ce = kfoldLoss(bagmodel);
accuracy=1-ce
%%
totalboostmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','TotalBoost','KFold',10);
ce = kfoldLoss(totalboostmodel);
accuracy=1-ce
%%
gentleboostmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','GentleBoost','KFold',10);
ce = kfoldLoss(gentleboostmodel);
accuracy=1-ce
%%
robustboostmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','RobustBoost','KFold',10);
ce = kfoldLoss(robustboostmodel);
accuracy=1-ce
%%
LPboostmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','LPBoost','KFold',10);
ce = kfoldLoss(LPboostmodel);
accuracy=1-ce
%%
RUSboostmodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','RUSBoost','KFold',10);
ce = kfoldLoss(RUSboostmodel);
accuracy=1-ce
%%
SubSpacemodel = fitcensemble(saliency_dense_features,imds.Labels,'Method','Subspace','KFold',10);
ce = kfoldLoss(SubSpacemodel);
accuracy=1-ce
%%
n=grp2idx(imds.Labels);
saliency_dense_mat=[double(saliency_dense_features),n];
%%
saliency_dense_features=double(saliency_dense_features);
%%
subspaceDimension = max(1, min(552, 1104 - 1));
classificationEnsemble = fitcensemble(...
    saliency_dense_features, ...
    n, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'discriminant', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', [1; 2]);
% Perform cross-validation
partitionedModel = crossval(classificationEnsemble, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')
%%
%% L-SVM
classificationSVM = fitcsvm(...
    saliency_dense_features, ...
    n, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
% Perform cross-validation
partitionedModel = crossval(classificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%% Q-SVM
classificationSVM = fitcsvm(...
    saliency_dense_features, ...
    n, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
% Perform cross-validation
partitionedModel = crossval(classificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%% cubic-SVM
classificationSVM = fitcsvm(...
    saliency_dense_features, ...
    n, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [1; 2]);
% Perform cross-validation
partitionedModel = crossval(classificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

%% 
only_saliency_mat=[saliency_features_mean,n];
nOfSelection=100;
population=20;
itrationMax=10000;
fun=@meandata;
[Selection , SelectionValue]=psoSelection(saliency_features_mean,nOfSelection,population,itrationMax,fun);
MinimizedData=saliency_features_mean(:,Selection);
disp('Selection---->')
disp(Selection)
disp('SelectionValue---->')
disp(SelectionValue)
GA_selected_saliency_mat=[MinimizedData,n];

%%
[trainZ,testZ] = ccaFuse(saliency_features_mean, deepFeatures, 'concat');
%%
rng(1); % For reproducibility
cvp = cvpartition(n,'Holdout',50)


Xtrain = saliency_features_mean(cvp.training,:);
ytrain = n(cvp.training,:);
Xtest  = saliency_features_mean(cvp.test,:);
ytest  = n(cvp.test,:);
%%
bin_lab=grp2idx(imds.Labels);
bin_lab(bin_lab==1)=0;
bin_lab(bin_lab==2)=1;
bin_lab(bin_lab==3)=2;

%%
[B,FitInfo] = lassoglm(saliency_dense_features,bin_lab,'binomial','CV',10);
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)]
yhat = glmval(coef,saliency_dense_features,'logit');
yhatBinom = (yhat>=0.5);
%Determine the accuracy of the predictions using a confusion matrix.
%%
acc = mean(bin_lab==yhatBinom)

[c_matrixp,Result]= confusion.getMatrix(bin_lab,yhatBinom);

disp('Getting Values')
Accuracy=Result.Accuracy
Error=Result.Error
Sensitivity=Result.Sensitivity
Specificity=Result.Specificity
Precision=Result.Precision
FalsePositiveRate=Result.FalsePositiveRate
F1_score=Result.F1_score
MatthewsCorrelationCoefficient=Result.MatthewsCorrelationCoefficient
Kappa=Result.Kappa
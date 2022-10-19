%% Naive bayesian 
naive_bayesian=fitcnb(...
        saliency_features_mean,imds.Labels, ...
        'DistributionNames', 'kernel', ...
        'KFold',10);
%%    
num_label=grp2idx(imds.Labels);
[validationPredictions, validationScores] = kfoldPredict(naive_bayesian);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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
%% cubic SVM
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
cubicSVM = fitcecoc(...
    saliency_features_mean,imds.Labels, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'KFold',10);
num_label=grp2idx(imds.Labels);
[validationPredictions, validationScores] = kfoldPredict(cubicSVM);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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

%% Fine-knn
knnfine = fitcknn(saliency_features_mean,imds.Labels,'NumNeighbors',1,...
    'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1,'KFold',10);
%%
[validationPredictions, validationScores] = kfoldPredict(knnfine);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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


%% weighted KNN
weightedKNN = fitcknn(...
    saliency_features_mean,imds.Labels, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true,...
    'KFold',10);
[validationPredictions, validationScores] = kfoldPredict(weightedKNN);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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
%% bagged enesmble
bagmodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','Bag','KFold',10);
%%
[validationPredictions, validationScores] = kfoldPredict(bagmodel);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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
%% subspace enesmble
SubSpacemodel = fitcensemble(saliency_features_mean,imds.Labels,'Method','Subspace','KFold',10);
%%
[validationPredictions, validationScores] = kfoldPredict(SubSpacemodel);
num_pred=grp2idx(validationPredictions);

[c_matrixp,Result]= confusion.getMatrix(num_label,num_pred);

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
%%
num_lab=grp2idx(imds.Labels);
saliency_mat=[saliency_features_mean,single(num_lab)];
%%
deepmodel=densenet201();

%%
imageSize = deepmodel.Layers(1).InputSize;
augmentedSet = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');

%%
featureLayer = 'fc1000';
deepFeatures = activations(deepmodel, augmentedSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
%%
deep_mat=[deepFeatures,single(num_lab)];
%%

for i=1:length(imds.Labels)

Image = readimage(imds,i);

[newfeatures]=newfeature(Image);

all_feat{i}=newfeatures;
end
%%
for i=1:length(imds.Labels)

Image = readimage(imds,i);

HOGfeatures{i}=hog_extractor(Image);
end
%%
hogfea_all=cell2mat(HOGfeatures');
%%
hog_mat=[hogfea_all,double(num_lab)];
%%
doubletypefea_all=cell2mat(all_feat');
%%
num_lab=grp2idx(imds.Labels);
doublefeat_mat=[doubletypefea_all,double(num_lab)];
%% PSO Selection
%% DEMO 2
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
%%
PSO_mat=[MinimizedData,single(num_lab)];
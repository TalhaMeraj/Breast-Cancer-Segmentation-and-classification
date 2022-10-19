%Igraph=createUnet();

%%
%Igraph=createUnet();
%%
dataSetDir = fullfile('G:\BreastCancerWorking\Datasets\INbreast\','Augmented_INbreast');
imageDir = fullfile(dataSetDir,'Train Images');
labelDir = fullfile(dataSetDir,'Train Masks');
%%

% Create an imageDatastore object to store the training images.

imds = imageDatastore(imageDir);
%%  Define the class names and their associated label IDs.

classNames = ["Tumor","Background"];
labelIDs   = [255 0];
%% Create a pixelLabelDatastore object to store the ground truth pixel labels for the training images.

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
%% Create the U-Net network.

InputSize = [512 512 1];
lgraph = createUnetNew(InputSize);
%%
 analyzeNetwork(lgraph)

%%
%Create a datastore for training the network.

ds = combine(imds,pxds);
%% Set training options.
checkpointPath='G:\BreastCancerWorking\MATLAB Codes\unet_checkpoints\augmented_unet\'
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',4, ...
    'MiniBatchSize',1,...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress',...
    'CheckpointPath',checkpointPath, ...
    'VerboseFrequency',10);

%% Train the network.

net = trainNetwork(ds,lgraph,options)
%%
load('net_checkpoint__34875__2022_05_21__09_45_35.mat','net')
%%
layers = net.Layers;
connections = net.Connections;
lgraph = createLgraphUsingConnections(layers,connections);
%%
net2 = trainNetwork(ds,lgraph,options);

%%
pxdsPredTrain = semanticseg(imds,net2,'MiniBatchSize',1);
%%
testimageDir = fullfile(dataSetDir,'Test Images');
testlabelDir = fullfile(dataSetDir,'Test Masks');
%%

% Create an imageDatastore object to store the training images.

imdsTest = imageDatastore(testimageDir);
%% Create a pixelLabelDatastore object to store the ground truth pixel labels for the training images.

pxdsTest = pixelLabelDatastore(testlabelDir,classNames,labelIDs);

%%
pxdsPredTest = semanticseg(imdsTest,net2,'MiniBatchSize',1);

%%
disp('training Data Results:')
trainingmetrics = evaluateSemanticSegmentation(pxdsPredTrain,ds);
%%
disp('testing Data Results:')
metrics = evaluateSemanticSegmentation(pxdsPredTest,pxdsTest);

%%
%% test on single image
I=imread('G:\BreastCancerWorking\Datasets\INbreast\processed_INbreast\Train Images\53580858.jpg');
[C,scores] = semanticseg(I,net,'MiniBatchSize',1);
%Overlay segmentation results on the image and display the results.
%%
B = labeloverlay(I, C);
figure
imshow(B)
%%
disp('training Data Metrices:')
trainingmetrics.DataSetMetrics
trainingmetrics.ClassMetrics
%%
disp('testing Data Metrices:')

metrics.DataSetMetrics
metrics.ClassMetrics


inputSize= [512 512 1];
numClasses= 2;
layers = [
    
    imageInputLayer(inputSize,'Name','Input-1')

        
    convolution2dLayer(3,128,'Name','Conv-1','DilationFactor',1,'Padding','same')
    convolution2dLayer(5,128,'Name','Conv-2','DilationFactor',1,'Padding','same')
    convolution2dLayer(1,128,'Name','Conv-3','DilationFactor',1,'Padding','same')

%     
%     groupedConvolution2dLayer(3,128,'channel-wise','Name','cwconv1')
%     
%     
%     transposedConv2dLayer(3,128,'Name','Upsample','Stride',1)
%     
%     additionLayer(2,'Name','add-1')

    batchNormalizationLayer('Name','BN-1')
    reluLayer('Name','relu-1')
    
    
%     batchNormalizationLayer('Name','BN-2')
%     reluLayer('Name','relu-2')
%     
%     additionLayer(2,'Name','relus1')

    
    
    
    convolution2dLayer(3,128,'Name','Conv-4','DilationFactor',2,'Padding','same')
    convolution2dLayer(5,128,'Name','Conv-5','DilationFactor',2,'Padding','same')
    convolution2dLayer(1,128,'Name','Conv-6','DilationFactor',2,'Padding','same')

    groupedConvolution2dLayer(1,128,'channel-wise','Name','cwconv2')

%     transposedConv2dLayer(3,128,'Name','Upsample2','Stride',1)

    
    depthConcatenationLayer(2,'Name','add-2')
    
    batchNormalizationLayer('Name','BN-3')
    reluLayer('Name','relu-3')
    

%     batchNormalizationLayer('Name','BN-7')
%     reluLayer('Name','relu-7')
    
    depthConcatenationLayer(2,'Name','relus1')

   
    
    
    convolution2dLayer(3,128,'Name','Conv-7','DilationFactor',4,'Padding','same')
    convolution2dLayer(5,128,'Name','Conv-8','DilationFactor',4,'Padding','same')
    convolution2dLayer(1,128,'Name','Conv-9','DilationFactor',4,'Padding','same')

    groupedConvolution2dLayer(9,128,'channel-wise','Name','cwconv3')    
    
    transposedConv2dLayer(9,128,'Name','Upsample3','Stride',1)

    depthConcatenationLayer(2,'Name','add-3')

    
    batchNormalizationLayer('Name','BN-8')
    reluLayer('Name','relu-8')
    
   
    
    convolution2dLayer(1,numClasses,'Name','Last-Conv-1')
    
%     transposedConv2dLayer(12,2,'Name','Upsample','Stride',1)    %additionLayer(3,'Name','add-4')
    
    %convolution2dLayer(1,numClasses,'Name','Last-Conv-2')
    
    
    softmaxLayer('Name','Softmax')
    
    
    
    pixelClassificationLayer('Name','PixelClassification')];
%%
lgraph = layerGraph(layers);

% Connect concatenation layers
lgraph = connectLayers(lgraph, 'Conv-1','add-1/in2');
% lgraph = connectLayers(lgraph, 'cwconv1','add-1/in3');

%% lgraph = connectLayers(lgraph, 'relu-4','add-1/in2');

lgraph = connectLayers(lgraph, 'Conv-4','add-2/in2');
% lgraph = connectLayers(lgraph, 'cwconv2','add-2/in3');

%%

lgraph = connectLayers(lgraph, 'Conv-7','add-3/in2');
% lgraph = connectLayers(lgraph, 'cwconv3','add-3/in3');
%%

 lgraph = connectLayers(lgraph, 'relu-1','relus1/in2');
%  lgraph = connectLayers(lgraph, 'relu-2','relus2/in2');
 %lgraph = connectLayers(lgraph, 'relu-8','relus3/in2');

%%
% lgraph = connectLayers(lgraph, 'relu-4','add-5/in2');
% lgraph = connectLayers(lgraph, 'relu-3','add-5/in3');

%%
%  lgraph = connectLayers(lgraph, 'Last-Conv-1','concat_1/in2');
%%
% %%
%  lgraph = connectLayers(lgraph, 'Conv-13','add-6/in2');
%  lgraph = connectLayers(lgraph, 'add-2','add-6/in3');
% %%
%  lgraph = connectLayers(lgraph, 'Conv-16','add-7/in2');
%  lgraph = connectLayers(lgraph, 'add-3','add-7/in3');
% %%
%  lgraph = connectLayers(lgraph, 'add-4','add-8/in2');
%  lgraph = connectLayers(lgraph, 'relu-4','add-8/in3');
%%
analyzeNetwork(lgraph)

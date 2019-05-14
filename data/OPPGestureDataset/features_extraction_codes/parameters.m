function param = parameters
%parameters file
%general parameters
param.classifier ='diagquadratic';

%k-value for knn
param.K = 3;

%parameters for svm
param.svmtrain='';
param.svmpredict='';

%parameters for gmm
param.covtype='diagonal';
param.mixtures=3;

%for feature extraction
param.FX.method='mean';
param.FX.window=15;
param.FX.step=8;

%Feature reduction
param.featureReduction.name = 'pca';
param.featureReduction.param = 50;

%%%handling missing value parameters
%mode: 'R' repeating last available value, 'D' deleting the whole vector
param.missingValue.mode = 'R';
param.missingValue.colthreshold = 0.90;


% setting various paths
addpath(genpath(cd));

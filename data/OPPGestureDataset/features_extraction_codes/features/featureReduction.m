function [fTrain,fTest]  = featureReduction(param,featureTrain,featureTest)
% This function is for performing feature reduction, like PCA etc.
% The inputs are:
%   param: the parameter file for setting various parameters, the type of
%   feature reduction technique has to be set here
%   featureTrain: The Training set for which the features have to be
%   reduced
%   featureTest: The test set for which the features have to be reduced
% The outputs are:
%   fTrain: training set after feature reduction
%   fTest: test set after feature reduction

switch param.name
    case 'pca'
        nAxe = param.param;
        [pc v] =  eigs(cov(featureTrain),[],nAxe);
        fTrain = featureTrain*pc;
        fTest = featureTest*pc;
    otherwise
        fTrain = featureTrain;
        fTest = featureTest;
end
        
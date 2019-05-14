function [Cllss,testLabel,Accuracy]=tanalyze(training,testing,ngroups,param)
% This function classifies the test data using the training set.
%--------------------------------------------------------------------------
% The inputs are:
%   training: it is the training set with the last columns being the class labels
%   test: it is the test set with the last columns being the actual class labels
%   ngroups: number of different types of classes. For example if ngroups=2,
%           then we are classifying test by two different ways
%   param: the script with all the parameter values
%--------------------------------------------------------------------------
% The outputs are:
%   Cllss: the predicted labels for test
%   testLabel: the actual labels of test
%   Accuracy: A structure of various accuracy measures
% -------------------------------------------------------------------------
% Constraint on the input data (for logical interpretation)
%   no. of columns of test data = no. of columns of training
%%    
if nargin<4
    param=parameters;
end

%sorting the training and test data as per the time stamps
    for i = 2:size(training,1)
        if training(i,1) < training(i-1,1)            
            training(i:end,1) = training(i:end,1) - (training(i,1) - training(i-1,1)) + (training(i-1,1) - training(i-2,1));
        end
    end
    for i = 2:size(testing,1)
        if testing(i,1) < testing(i-1,1)            
            testing(i:end,1) = testing(i:end,1) - (testing(i,1) - testing(i-1,1)) + (testing(i-1,1) - testing(i-2,1));
        end
    end
    
% Handle missing values
    [training testing] = missingValueHandler(training,testing,param.missingValue);
    
% Separating various parts of the data 
    szData=size(training);
    szTest=size(testing);
    
    train=training(:,1:szData(2)-ngroups);
% Feature extraction for training data
    trainFeature=featureExtraction(train,param.FX,1);
    
    trainLabel=training(:,szData(2)-ngroups+1:szData(2));
    % Recalculating groups based on FX
    for i=1:1:size(trainLabel,2)
        trainFeatureLabel(:,i)=featureExtraction(trainLabel(:,i),param.FX,2);
    end
              
    test=testing(:,1:szTest(2)-ngroups);
% Feature extraction for test data
    testFeature=featureExtraction(test,param.FX,1);
    testLabel=testing(:,szTest(2)-ngroups+1:szTest(2));
    for i=1:1:size(trainLabel,2)
        testFeatureLabel(:,i)=featureExtraction(testLabel(:,i),param.FX,2);        
    end
 %% Classification
    ROC = cell(1,ngroups);
    f = cell(1,ngroups);
    threshold = cell(1,ngroups);
     switch param.classifier
    % LDA clasiifier
         case 'diaglinear'
            Cls=zeros(size(testFeature,1),ngroups);    
            for col=1:1:ngroups
                [threshold{col}, Cls(:,col),f{col},ROC{col}]=classifyAndReject(trainFeature,trainFeatureLabel(:,col),unique(trainLabel(:,col)),...
                    testFeature,testFeatureLabel(:,col),'diaglinear');
                Accuracy(col).threshold=threshold{col};
                Accuracy(col).threshold_limits=f{col};
                Accuracy(col).ROC = ROC{col};
            end
    % QDA classifier       
        case 'diagquadratic'
            Cls=zeros(size(testFeature,1),ngroups);
            for col=1:1:ngroups
                [threshold{col}, Cls(:,col),f{col},ROC{col}]=classifyAndReject(trainFeature,trainFeatureLabel(:,col),unique(trainLabel(:,col)),...
                    testFeature,testFeatureLabel(:,col),'diagquadratic');                
                Accuracy(col).threshold=threshold{col};
                Accuracy(col).threshold_limits=f{col};
                Accuracy(col).ROC = ROC{col};
            end
            
    % knn classifier
        case 'knn'
            Cls=zeros(size(testFeature,1),ngroups);
            for col=1:1:ngroups
                Cls(:,col)=knn(testFeature,trainFeature,trainFeatureLabel(:,col),param.K);
            end
    % ncc classifier
        case 'ncc'
            for col=1:1:ngroups
                [threshold{col}, Cls(:,col),f{col},ROC{col}]=classifyAndReject(trainFeature,trainFeatureLabel(:,col),unique(trainLabel(:,col)),...
                    testFeature,testFeatureLabel(:,col),'ncc');
                Accuracy(col).threshold=threshold{col};
                Accuracy(col).threshold_limits=f{col};
                Accuracy(col).ROC = ROC{col};
            end
    end
%expanding predicted labels to original size
    if strcmp(param.FX.method,'raw')
        Cllss=Cls;
    else
    for i=1:1:size(Cls,2)
        Cllss(:,i)=expandingLabels(Cls(:,i),param.FX.window,size(testLabel,1),param.FX.step);
    end
    end
%% Accuracy
    for i=1:1:ngroups
        [Accuracy(i).F, Accuracy(i).ward]=clsAccuracy(testLabel(:,i),Cllss(:,i),test);
    end
end
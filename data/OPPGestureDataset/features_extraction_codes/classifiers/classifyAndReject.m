function [threshold, Class, f,ROC] = nullrejection3(trainFeature,trainFeatureLabel,labels,...
    testFeature,testFeatureLabel,classifier)
% This function classifies data and generates the most appropriate threshold for rejecting
% Null data (based on F measure), for the LDA, QDA and NCC classifeirs and also uses this
% threshold for the classification. The threshold is chosen such that it
% gives the best accuracy of classification based on the F-measures
%--------------------------------------------------------------------------
%The inputs are:
%   trainFeature: the training set after feature extraction
%   Truth: the class labels of the training set
%   testFeature: the testing set after feature extraction
%   trainFeatureLabel: The actual labels of testFeature
%   classifier: the classifier to be used; LDA, QDA, NCC
%   FXparam: the parameters file with the various parameters required
%   testFeatureLabel: class labels of test set
%The outputs are:
%   threshold: the most appropriate threshold
%   Class: the array of predicted labels
%   f: structure of F-measures
%   ROC: structure containing TP rate and FN rate on test set.
%--------------------------------------------------------------------------

classes = labels;
classes = classes(2:end);%remove null
nClass = length(classes);
if any(cell2mat(strfind({'diaglinear','diagquadratic'},classifier)))
    %[C,~,p] = GausianClassify(trainFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0),classifier);
    [C,xunused,p] = GausianClassify(trainFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0),classifier);
elseif strcmp('ncc',classifier)
    %[C,~,p]=nccClassify(trainFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0));
    [C,xunused,p]=nccClassify(trainFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0));
end

f=zeros(nClass,12);
threshold =zeros(1,nClass);
threshClass = cell(1,nClass);
for iClass = 1:nClass
    pclass = p(:,iClass);
    k1 = mean(pclass(trainFeatureLabel ~= classes(iClass)));    
    k2 = mean(pclass(trainFeatureLabel == classes(iClass)));
    m2 = min(pclass(trainFeatureLabel == classes(iClass)));
    s = sort([m2, k1,k2]);
    
    range1 = [s(1) s(2) s(3) s(3)+s(3)-s(2)];
    t = [linspace(range1(1),range1(2),20) linspace(range1(2),range1(3),30) linspace(range1(3),range1(4),20)];
    threshClass{iClass} = t;
    
    TPr = zeros(1,length(t));
    FNr = zeros(1,length(t));
    tr = trainFeatureLabel;
    tr(tr~=classes(iClass)) = 0;
    for ii=1:length(t);
        CR = C;
        CR(p(:,iClass)<t(ii))=classes(iClass);
        CR(p(:,iClass)>=t(ii))= 0;
        %measuring accuracy
        conf(2,2) = sum(tr==classes(iClass) & CR == classes(iClass));%TP
        conf(2,1) = sum(tr==classes(iClass) & CR ~= classes(iClass));%FN
        conf(1,2) = sum(tr~=classes(iClass) & CR == classes(iClass));%FP
        conf(1,1) = sum(tr~=classes(iClass) & CR ~= classes(iClass));%TN
        f(iClass,ii) = measures('F',conf,'two');
        TPr(ii) = conf(2,2)/sum(conf(2,:));
        FNr(ii) = conf(1,2)/sum(conf(1,:));
    end
    
    %[~,index]=max(f(iClass,:));
    [xunused,index]=max(f(iClass,:));
    if sum(f(iClass,:)) == 0
        index = 5;
    end
    threshold(iClass)=t(index);
end
if any(cell2mat(strfind({'diaglinear','diagquadratic'},classifier)))
    %[Class,~,pt]=GausianClassify(testFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0),classifier);
    [Class,xunused,pt]=GausianClassify(testFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0),classifier);
else strcmp('ncc',classifier)
    %[Class,~,pt]=nccClassify(testFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0));
    [Class,xunused,pt]=nccClassify(testFeature,trainFeature(trainFeatureLabel~=0,:),trainFeatureLabel(trainFeatureLabel~=0));
end

%ROC for test
ROC = cell(1,nClass);
for iClass = 1:nClass
    pclass = pt(:,iClass);
    k1 = mean(pclass(testFeatureLabel ~= classes(iClass)));
    k2 = mean(pclass(testFeatureLabel == classes(iClass)));
    m2 = min(pclass(testFeatureLabel == classes(iClass)));
    s = sort([m2, k1,k2]);
    range1 = [s(1) s(2) s(3) s(3)+s(3)-s(2)];
    t = [linspace(range1(1),range1(2),20) linspace(range1(2),range1(3),30) linspace(range1(3),range1(4),20)];
    
    TPr = zeros(1,length(t));
    FNr = zeros(1,length(t));
    ts = testFeatureLabel;
    ts(ts~=classes(iClass)) = 0;
    for ii=1:length(t);
        CR = Class;
        CR(pt(:,iClass)<t(ii))=classes(iClass);
        CR(pt(:,iClass)>=t(ii))= 0;
        %measuring accuracy
        conf(2,2) = sum(ts==classes(iClass) & CR == classes(iClass));%TP
        conf(2,1) = sum(ts==classes(iClass) & CR ~= classes(iClass));%FN
        conf(1,2) = sum(ts~=classes(iClass) & CR == classes(iClass));%FP
        conf(1,1) = sum(ts~=classes(iClass) & CR ~= classes(iClass));%TN
        TPr(ii) = conf(2,2)/sum(conf(2,:));
        FNr(ii) = conf(1,2)/sum(conf(1,:));
    end
    ROC{iClass}.TPr = TPr;
    ROC{iClass}.FNr = FNr;
end


for iClass = 1:nClass
    in = Class == classes(iClass);
    z = pt(:,iClass)>threshold(iClass);
    Class(in & z)=0;
end
%% this script classifies and generates the "ward" accuracy measures
% This is the master script
%--------------------------------------------------------------------------
% the structure of the results is:
% the main output is the variable resultmean which is a structure.
%      resultmean.acc is the structure array of accuracy measures
%      resultmean.pred is the structure array of predicted labels
%      resultmean.actCls is the structure array of actual labels
% -------------------------------------------------------------------------
% resultmean.acc  
%   resultmean.acc is a (nSubjects x nClassifiers) array. The row indicates
%   the Subject and the columns indicate the classifier in the order: 
%           LDA, QDA, KNN1, KNN3, NCC
%   Each element is a structure array with 2 elements
%       the first is for various prediction measures for locomotion
%       the second is for various prediction measures for gestures
%
%   The code uses normMact, normMnull and diagavg parts of the element
% -------------------------------------------------------------------------
% resultmean.pred
%   resultmean.pred is a(nSubjects x nClassifiers) array. The row indicates
%   the Subject and the columns indicate the classifier in the order: 
%           LDA, QDA, KNN1, KNN3, NCC
%   Each element is a structure array with 2 elements
%       the first is for various accuracy measures for locomotion
%       the second is for various accuracy measures for gestures
% -------------------------------------------------------------------------        
% resultmean.actCls is a (nSubjects x 1) array with each element being an
% array of the actual class labels
% -------------------------------------------------------------------------
%  The script also generates bar plots of various measures using methods
%  developed by Jamie Ward et al. An excel file is also generated which has
%  the values of the various accuracy measures.
% -------------------------------------------------------------------------
% Excel file
% The columns are Null Activity, Sensitivity, Precision, FP,
%   normMact- weighted avg. of f measures of conf matrix without null class
%   normMnull- weighted avg. of f measures of conf matrix with null class
%   diagavg- sum of diag. of conf matrix/sum of all elements of confmatrix
% The rows are LDA, QDA, KNN1, KNN3, NCC grouped by subjects i.e.
%   row1-row5: Subj1,   row6-row10: Subj2.....
%--------------------------------------------------------------------------

%% Classsification
tic
%Subject 1

% only motion jacket
% test1 = test1(:,[1 38:82 end-1 end]);
% test2 = test2(:,[1 38:82 end-1 end]);
% test3 = test3(:,[1 38:82 end-1 end]);
% training1 = training1(:,[1 38:82 end-1 end]);
% training2 = training2(:,[1 38:82 end-1 end]);
% training3 = training3(:,[1 38:82 end-1 end]);
% test2 = [test2;test3];
% training2 = [training2;training3];

sprintf('subject 1\n')
param=parameters;
if strcmp('meanVar',param.FX.method)
    param.featureReduction.param=110;
end
param.classifier='diaglinear';fprintf(param.classifier);
[resultmean.pred{1,1},resultmean.actCls{1,1},resultmean.acc{1,1}]=tanalyze(training1,test1,2,param);
param.classifier='diagquadratic';fprintf(param.classifier);
%[resultmean.pred{1,2},~,resultmean.acc{1,2}]=tanalyze(training1,test1,2,param);
[resultmean.pred{1,2},xunused,resultmean.acc{1,2}]=tanalyze(training1,test1,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=1;
%[resultmean.pred{1,3},~,resultmean.acc{1,3}]=tanalyze(training1,test1,2,param);
[resultmean.pred{1,3},xunused,resultmean.acc{1,3}]=tanalyze(training1,test1,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=3;
%[resultmean.pred{1,4},~,resultmean.acc{1,4}]=tanalyze(training1,test1,2,param);
[resultmean.pred{1,4},xunused,resultmean.acc{1,4}]=tanalyze(training1,test1,2,param);
param.classifier='ncc';fprintf(param.classifier);
%[resultmean.pred{1,5},~,resultmean.acc{1,5}]=tanalyze(training1,test1,2,param);
[resultmean.pred{1,5},xunused,resultmean.acc{1,5}]=tanalyze(training1,test1,2,param);

%Subject 2
sprintf('subject 2\n')
param=parameters;
if strcmp('meanVar',param.FX.method)
    param.featureReduction.param=113;
end
param.classifier='diaglinear';fprintf(param.classifier);
[resultmean.pred{2,1},resultmean.actCls{2,1},resultmean.acc{2,1}]=tanalyze(training2,test2,2,param);
param.classifier='diagquadratic';fprintf(param.classifier);
%[resultmean.pred{2,2},~,resultmean.acc{2,2}]=tanalyze(training2,test2,2,param);
[resultmean.pred{2,2},xunused,resultmean.acc{2,2}]=tanalyze(training2,test2,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=1;
%[resultmean.pred{2,3},~,resultmean.acc{2,3}]=tanalyze(training2,test2,2,param);
[resultmean.pred{2,3},xunused,resultmean.acc{2,3}]=tanalyze(training2,test2,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=3;
%[resultmean.pred{2,4},~,resultmean.acc{2,4}]=tanalyze(training2,test2,2,param);
[resultmean.pred{2,4},xunused,resultmean.acc{2,4}]=tanalyze(training2,test2,2,param);
param.classifier='ncc';fprintf(param.classifier);
%[resultmean.pred{2,5},~,resultmean.acc{2,5}]=tanalyze(training2,test2,2,param);
[resultmean.pred{2,5},xunused,resultmean.acc{2,5}]=tanalyze(training2,test2,2,param);

%Subject3
sprintf('subject 3\n')
param=parameters;
if strcmp('meanVar',param.FX.method)
    param.featureReduction.param=113;
end
param.classifier='diaglinear';fprintf(param.classifier);
[resultmean.pred{3,1},resultmean.actCls{3,1},resultmean.acc{3,1}]=tanalyze(training3,test3,2,param);
param.classifier='diagquadratic';fprintf(param.classifier);
%[resultmean.pred{3,2},~,resultmean.acc{3,2}]=tanalyze(training3,test3,2,param);
[resultmean.pred{3,2},xunused,resultmean.acc{3,2}]=tanalyze(training3,test3,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=1;
%[resultmean.pred{3,3},~,resultmean.acc{3,3}]=tanalyze(training3,test3,2,param);
[resultmean.pred{3,3},xunused,resultmean.acc{3,3}]=tanalyze(training3,test3,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=3;
%[resultmean.pred{3,4},~,resultmean.acc{3,4}]=tanalyze(training3,test3,2,param);
[resultmean.pred{3,4},xunused,resultmean.acc{3,4}]=tanalyze(training3,test3,2,param);
param.classifier='ncc';fprintf(param.classifier);
%[resultmean.pred{3,5},~,resultmean.acc{3,5}]=tanalyze(training3,test3,2,param);
[resultmean.pred{3,5},xunused,resultmean.acc{3,5}]=tanalyze(training3,test3,2,param);

%Subject4
sprintf('subject 4\n')
param=parameters;
if strcmp('meanVar',param.FX.method)
    param.featureReduction.param=45;
end
param.classifier='diaglinear';fprintf(param.classifier);
[resultmean.pred{4,1},resultmean.actCls{4,1},resultmean.acc{4,1}]=tanalyze(training4,test4,2,param);
param.classifier='diagquadratic';fprintf(param.classifier);
%[resultmean.pred{4,2},~,resultmean.acc{4,2}]=tanalyze(training4,test4,2,param);
[resultmean.pred{4,2},xunused,resultmean.acc{4,2}]=tanalyze(training4,test4,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=1;
%[resultmean.pred{4,3},~,resultmean.acc{4,3}]=tanalyze(training4,test4,2,param);
[resultmean.pred{4,3},xunused,resultmean.acc{4,3}]=tanalyze(training4,test4,2,param);
param.classifier='knn';fprintf(param.classifier);
param.K=3;
%[resultmean.pred{4,4},~,resultmean.acc{4,4}]=tanalyze(training4,test4,2,param);
[resultmean.pred{4,4},xunused,resultmean.acc{4,4}]=tanalyze(training4,test4,2,param);
param.classifier='ncc';fprintf(param.classifier);
%[resultmean.pred{4,5},~,resultmean.acc{4,5}]=tanalyze(training4,test4,2,param);
[resultmean.pred{4,5},xunused,resultmean.acc{4,5}]=tanalyze(training4,test4,2,param);

save('results.mat','resultmean');
%% Arranging the data and plotting accuracies
nSubj=size(resultmean.acc,1);
nClassifiers=size(resultmean.acc,2);

label={'t1DL', 't1DQ', 't1KNN1', 't1KNN3', 't1NCC';'t2DL', 't2DQ', 't2KNN1', 't2KNN3', 't2NCC';'tDL', 't3DQ', 't3KNN1', 't3KNN3', 't3NCC';'t4DL', 't4DQ', 't4KNN1', 't4KNN3', 't4NCC'};
data1=[];
data2=data1;
for iSubj=1:nSubj
    for jClass=1:nClassifiers
        data1{iSubj,jClass} = resultmean.acc{iSubj,jClass}(1,1).ward.t;
        data2{iSubj,jClass} = resultmean.acc{iSubj,jClass}(1,2).ward.t;
    end
end

%%
% [data1 label1]=resultsArrange(4,5,'t1DL',AccuracyDLmean1(1).ward.t,'t1DQ',AccuracyDQmean1(1).ward.t,'t1KNN1',AccuracyKNN1mean1(1).ward.t,'t1KNN3',AccuracyKNN3mean1(1).ward.t,'t1NCC',AccuracyNCCmean1(1).ward.t,'t2DL',AccuracyDLmean2(1).ward.t,'t2DQ',AccuracyDQmean2(1).ward.t,'t2KNN1',AccuracyKNN1mean2(1).ward.t,'t2KNN3',AccuracyKNN3mean2(1).ward.t,'t2NCC',AccuracyNCCmean2(1).ward.t,'t3DL',AccuracyDLmean3(1).ward.t,'t3DQ',AccuracyDQmean3(1).ward.t,'t3KNN1',AccuracyKNN1mean3(1).ward.t,'t3KNN3',AccuracyKNN3mean3(1).ward.t,'t3NCC',AccuracyNCCmean3(1).ward.t,'t4DL',AccuracyDLmean4(1).ward.t,'t4DQ',AccuracyDQmean4(1).ward.t,'t4KNN1',AccuracyKNN1mean4(1).ward.t,'t4KNN3',AccuracyKNN3mean4(1).ward.t,'t4NCC',AccuracyNCCmean4(1).ward.t);
% [data2 label2]=resultsArrange(4,5,'t1DL',AccuracyDLmean1(2).ward.t,'t1DQ',AccuracyDQmean1(2).ward.t,'t1KNN1',AccuracyKNN1mean1(2).ward.t,'t1KNN3',AccuracyKNN3mean1(2).ward.t,'t1NCC',AccuracyNCCmean1(2).ward.t,'t2DL',AccuracyDLmean2(2).ward.t,'t2DQ',AccuracyDQmean2(2).ward.t,'t2KNN1',AccuracyKNN1mean2(2).ward.t,'t2KNN3',AccuracyKNN3mean2(2).ward.t,'t2NCC',AccuracyNCCmean2(2).ward.t,'t3DL',AccuracyDLmean3(2).ward.t,'t3DQ',AccuracyDQmean3(2).ward.t,'t3KNN1',AccuracyKNN1mean3(2).ward.t,'t3KNN3',AccuracyKNN3mean3(2).ward.t,'t3NCC',AccuracyNCCmean3(2).ward.t,'t4DL',AccuracyDLmean4(2).ward.t,'t4DQ',AccuracyDQmean4(2).ward.t,'t4KNN1',AccuracyKNN1mean4(2).ward.t,'t4KNN3',AccuracyKNN3mean4(2).ward.t,'t4NCC',AccuracyNCCmean4(2).ward.t);
[handle1,measuresLoco]=wardbars(label,data1);
title('Locomotion','FontWeight','bold','FontSize',12);
[handle2,measuresGest]=wardbars(label,data2);
title('Gestures','FontWeight','bold','FontSize',12);
save('measuresLocoII.mat','measuresLoco');
save('measuresGestII.mat','measuresGest');
%% Exporting various measures to xls file
% Change the file location and name to avoid overwrite
matloco=res2mat(measuresLoco,resultmean.acc,'loco');
matgest=res2mat(measuresGest,resultmean.acc,'gest');
xlswrite('benchmark.xls',matloco,'Sheet1','C3');
xlswrite('benchmark.xls',matgest,'Sheet2','C3');
toc
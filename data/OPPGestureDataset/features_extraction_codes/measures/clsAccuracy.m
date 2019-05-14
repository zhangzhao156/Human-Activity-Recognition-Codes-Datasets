function [accuracy,accuracy1] = clsAccuracy(ActCls, Cllss, Test)
    %   inputs are ActCls = Actual Classes, Cllss = Predicted Class 
    %   outpus are different accuracy measures of the classification
        
    % f - measures
    conf=confusionmat(ActCls,Cllss);
    [accuracy.fmeasures accuracy.normMact accuracy.normMnull]=measures('F',conf);
    accuracy.diagavg=measures('A',conf);
    
    % ward accuracies
    t=Test(:,1);
    ground=zeros(size(ActCls,1),3);
    eval=zeros(size(Cllss,1),3);
    
    projectedCllss=separate(Cllss,1);
    projectedActCls=separate(ActCls,1);
    
    ground(:,1)=t;
    eval(:,1)=t;
    
    % the first 2 columns are time intervals
    ground(1:end-1,2)=t(2:end);
    ground(end,2)=t(end)+t(2)-t(1);
    ground(:,3)=projectedActCls;
    eval(1:end-1,2)=t(2:end);
    eval(end,2)=t(end)+t(2)-t(1);
    eval(:,3)=projectedCllss;
  
    [accuracy1.t,accuracy1.s,accuracy1.e,accuracy1.meanLenOU]=mset(eval,ground,length(unique(projectedActCls)));
end
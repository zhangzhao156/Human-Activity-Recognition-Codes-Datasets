% Script which plots the ROC curves for the different thresholds for the
% classifiers LDA, QDA and NCC.
% There are 4 classes in Locomotion and 16 classes in Gestures.
% The output is 8 plots for all the 4 subjects

for iSubj=1:size(resultmean.acc,1);

% Roc curves for locomotion
figure;
for i=1:4
subplot(2,2,i)
plot([0 1],[0 1],'--c');
hold on;
h1=plot(resultmean.acc{iSubj,1}(1,1).ROC{i}.FNr,resultmean.acc{iSubj,1}(1,1).ROC{i}.TPr,'b');
% plot(AccuracyDLmean4(1).ROC{i}.FNr,AccuracyDLmean4(1).ROC{i}.TPr,'b');
h2=plot(resultmean.acc{iSubj,2}(1,1).ROC{i}.FNr,resultmean.acc{iSubj,2}(1,1).ROC{i}.TPr,'r');
h3=plot(resultmean.acc{iSubj,5}(1,1).ROC{i}.FNr,resultmean.acc{iSubj,5}(1,1).ROC{i}.TPr,'g');
% plot(AccuracyDQmean4(1).ROC{i}.FNr,AccuracyDQmean4(1).ROC{i}.TPr,'r');
% plot(AccuracyNCCmean4(1).ROC{i}.FNr,AccuracyNCCmean4(1).ROC{i}.TPr,'m');
t=['Subject ' mat2str(iSubj) ' - Locomotion'];
title(gca,t);
axis square;
end

linelegend={'NCC   ';'QDA   ';'LDA   '};
legend([h1;h2;h3],linelegend(end:-1:1),-1);

% ROC curves for Gestures
labels = [506616   
506617  
504616   
504617   
506620   
504620   
506605  
504605  
506619   
504619   
506611   
504611   
506608   
504608   
508612   
507621   
505606   
];
strLabels={...
    'Open Door 1'...
   ,'Open Door 2'...
   ,'Close Door 1'...
   ,'Close Door 2'...
   ,'Open Fridge'...
   ,'Close Fridge'...
   ,'Open Dishwasher'...
   ,'Close Dishwasher'...
   ,'Open Drawer 1'...
   ,'Close Drawer 1'...
   ,'Open Drawer 2'...
   ,'Close Drawer 2'...
   ,'Open Drawer 3'...
   ,'Close Drawer 3'...
   ,'Clean Table'...
   ,'Drink Cup'...
   ,'Toggle Switch'
};

[s ts] = sort(labels);

figure;
for ii=1:17
subplot(3,6,ii)
i = find(ts==ii);
plot([0 1],[0 1],'--c');
hold on;
h1=plot(resultmean.acc{iSubj,1}(1,2).ROC{i}.FNr,resultmean.acc{iSubj,1}(1,2).ROC{i}.TPr,'b');
% plot(AccuracyDLmean4(2).ROC{i}.FNr,AccuracyDLmean4(2).ROC{i}.TPr,'b');
h2=plot(resultmean.acc{iSubj,2}(1,2).ROC{i}.FNr,resultmean.acc{iSubj,2}(1,2).ROC{i}.TPr,'r');
h3=plot(resultmean.acc{iSubj,5}(1,2).ROC{i}.FNr,resultmean.acc{iSubj,5}(1,2).ROC{i}.TPr,'g');
% plot(AccuracyDQmean4(2).ROC{i}.FNr,AccuracyDQmean4(2).ROC{i}.TPr,'r');
% plot(AccuracyNCCmean4(2).ROC{i}.FNr,AccuracyNCCmean4(2).ROC{i}.TPr,'m');
t=strLabels{ii};%['Subject ' mat2str(iSubj) ' - Gestures'];
title(gca,t);
axis square;
end
linelegend={'NCC   ';'QDA   ';'LDA   '};
legend([h1;h2;h3],linelegend(end:-1:1),-1);

end
function table = res2mat(resW,resF,groupName)
% The function outputs an array containing various accuracy measures
% The output can be expoted as a ".xls" file using the function "xlswrite"
% The inputs are:
%     resW: the array of Ward accuracy measures
%     resF: the array of F-measures
%     groupName: 'Loco' or 'Gest' for choosing the correct group

nSubj=size(resW,1);
nCls=size(resW,2);
k=0;
% Choosing the right group
switch groupName
    case 'loco'
        t=1;
    case 'gest'
        t=2;
end
% Building the output array
table=zeros(nSubj*nCls,7); %we have 7 diff measures
for i=1:nSubj
    for j=1:nCls
        table(j+k,1)=resW(i,j).NULLactivity;
        table(j+k,2)=resW(i,j).recall;
        table(j+k,3)=resW(i,j).precision;
        table(j+k,4)=resW(i,j).fp;
        table(j+k,5)=resF{i,j}(1,t).F.normMact*100;
        table(j+k,6)=resF{i,j}(1,t).F.normMnull*100;
        table(j+k,7)=resF{i,j}(1,t).F.diagavg.avgDiag*100;
    end
    k=k+nCls;
end
end
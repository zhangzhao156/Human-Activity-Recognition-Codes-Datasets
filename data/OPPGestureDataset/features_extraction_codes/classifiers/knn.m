function class=knn(DTest,DTrain,Group,k)
% This function id for the KNN classifier
% The inputs are:
%   DTest: the test data set
%   DTrain: the training data set
%   Group: the actual labels of the training data set
%   k: the no. of nearest neighbours to be taken for classification
% The output is
%   class: the predicted labels for the test data set

    szTest=size(DTest);   
    class=zeros(szTest(1),1);
    D = zeros(size(DTrain));
    for i=1:1:szTest(1)
        for j=1:1:szTest(2)
            D(:,j) = (DTrain(:,j)-DTest(i,j)).^2;
        end
        Distance = sum(D,2);
        %[~,index]=sort(Distance,'ascend');
        [xunused,index]=sort(Distance,'ascend');
        kn=Group(index(1:k));
        class(i,1)=mode(kn);
    end
end
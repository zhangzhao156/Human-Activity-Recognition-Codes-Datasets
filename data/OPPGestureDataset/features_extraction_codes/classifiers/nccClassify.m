function [ c post p ]  = nccClassify2(  test,train,group)
%NCCCLASSIFY (Nearest Centroid Classifer)
%Input:
% test:pxf
% train:pxf
% group:classes
%Output:
% c: predicted classes
% post: normalized (1/distance) for all the classes
% p: distance to the nearest center (far away to a center -> less
% value)

classes = unique(group);
nClass = length(classes);
%training
obj = cell(1,nClass);
for iClass=1:nClass
    z = group == classes(iClass);
    obj{iClass} = mean(train(z,:));
end
%Classification
dis = zeros(size(test,1),nClass);
for iClass = 1:nClass
    m = sum((test - repmat(obj{iClass},[size(test,1) 1])).^2,2);
    dis(:,iClass) = m;
end
[p,minClass] = min(dis,[],2);
post = 1./dis;
post = post./repmat(sum(post,2),[1,nClass]);
p = dis;
c = classes(minClass);
end


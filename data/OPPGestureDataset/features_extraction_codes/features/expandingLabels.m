function [ expandedLabels ] = expandingLabels( windowedlabels, win, lenData,step)
%EXPANDINGLABELS, expand the labels to have the same size of the data
%Inputs:
% windowedlabels: the labels which are already compressed (by windowingLabels)
% win: length of voting window
% lenData: the desired length of the labels
%output:
% expandedLabels: vector of uncompressed or expanded labels
expandedLabels = zeros(lenData,1);
s = length(windowedlabels);
%expandedLabels(1:win)=windowedlabels(1);
k = 1;%win+1;
for i= 1:s
    expandedLabels(k:k+step-1) = windowedlabels(i);
    k = k + step;
end
expandedLabels(k:end) = windowedlabels(s);
end


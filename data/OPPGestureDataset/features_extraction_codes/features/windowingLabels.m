function [ labels ] = windowingLabels( labelStream, win,step)
%WINDOWINGLABELS takes the majority of the labels in a window of size win.
%There is no overlap.
%input:
% labelStream: vector of labels
% win: window size
%output:
% labels:windowed labels
s = length(labelStream);
labels = zeros(s,1);
k = 1;
for i = 1:step:s
    labels(k) = mode(labelStream(i:min(i+win-1,s)));
    k=k+1;
end
labels(k-1:end) = [];
end
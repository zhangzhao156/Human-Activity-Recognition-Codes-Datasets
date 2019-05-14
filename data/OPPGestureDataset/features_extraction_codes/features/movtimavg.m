function output=movtimavg(input,n,step,includeVar)
% This function performs the moving time averge of the input stream with a  
% given window length and step size. Variance can also be output along with
% mean.
%--------------------------------------------------------------------------
% The inputs are:
%   input: the input data stream
%   n: length of moving window
%   step: step length
%   invludeVar: set as 1 if you want to output variance also as a feature
% The output is the moving time average (with/without variance)

    input = input(:,2:end);
    s=size(input,1);
    output=zeros(s,size(input,2));
    outputVar=zeros(s,size(input,2));
    k=1;
    for i=1:step:s
        output(k,:)=mean((input(i:min(i+n-1,s),:)),1);
        outputVar(k,:)=var((input(i:min(i+n-1,s),:)));	
        k=k+1;
    end
    output(k-1:end,:) = [];
    outputVar(k-1:end,:) = [];
    if nargin>3
       if includeVar
         output = [output,outputVar];
       end
    end
end

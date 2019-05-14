function [training,test]=tarrange(ntrain,varargin)
%   this function concatanates and builds the test and training data
%   inputs are no of training sets, training variables, test variables
%   input the training variables before the test variables

    training=varargin(1);
    training=training{1,1};
    for i=2:1:ntrain
        a=varargin(i);
        a=a{1,1};
        training=[training;a];
    end
    
    test=varargin(ntrain+1);
    test=test{1,1};
    for i=ntrain+2:1:size(varargin,2)
        a=varargin(i);
        a=a{1,1};
        test=[test;a];
    end
    
    %sorting the training and test data as per the time stamps
    %[~,index1]=sort(training(:,1));
    %training=training(index1,:);
    %[~,index2]=sort(test(:,1));
    %test=test(index2,:);

%sorting the training and test data as per the time stamps
    for i = 2:size(training,1)
        if training(i,1) < training(i-1,1)            
            training(i:end,1) = training(i:end,1) - (training(i,1) - training(i-1,1)) + (training(i-1,1) - training(i-2,1));
        end
    end
    for i = 2:size(test,1)
        if test(i,1) < test(i-1,1)            
            test(i:end,1) = test(i:end,1) - (test(i,1) - test(i-1,1)) + (test(i-1,1) - test(i-2,1));
        end
    end
    
end

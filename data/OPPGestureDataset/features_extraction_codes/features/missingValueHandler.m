function [training,test] = missingValueHandler(training,test,param)
% Omitting sensors (columns) which have NaN more than threshold, column 1 = time
nanCol = sum(isnan(training),1)>(size(training,1)*param.colthreshold);
training(:,nanCol) = [];
test(:,nanCol) = [];
% For other nan values do a technique to replace or remove them.    
switch param.mode
        case 'D'
            %deleting rows which have NaN
            %%%%%%%%%%%%%%%%%%%%%REMOVE THE SAME COLOMNS: CHECK AGAIN%%%%
            training=training(sum(isnan(training),2)==0,:);       
            test=test(sum(isnan(test),2)==0,:);
        case 'R'
            %replacing NaN with the previous value instead of deleting
            % takes about 30s to compute  
            %Replacing first row with NaNs with 0
            for j=1:size(training,2)
                if isnan(training(1,j))
                    training(1,j)=0;
                end
            end
            %replacing other NaNs with previous value
            for i=2:size(training,1)
                for j=1:1:size(training,2)
                    if isnan(training(i,j))
                        training(i,j)=training(i-1,j);
                    end
                end
            end
            %removing the col with var=0 from training and test
            vTr = var(training);
            remcoltr = vTr==0;%The same value
            others = 1:size(training,2);
            others = setdiff(others,remcoltr);
            training = training(:,others);
            test = test(:,others);
            %calculating mean of training to use for the first row of test
            %with NaNs
            mtr = mean(training);
            for j=1:1:size(test,2)
                if isnan(test(1,j))
                    test(1,j)=mtr(j);
                end
            end
            %Repeating the NaNs with previous value in test set
            for i=2:size(test,1)
                for j=1:1:size(test,2)
                    if isnan(test(i,j))==1
                        test(i,j)=test(i-1,j);
                    end
                end
            end
end
end
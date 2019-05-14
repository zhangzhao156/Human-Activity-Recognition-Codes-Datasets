function [f normMact normMnull] = measures1(method,input,mode)
% Measuring Accuracy
% Hesam Sagha 3 Jun 2011
% Inputs
%  method: 'F'-> F-measure, 'A' -> accuracy (diag/sum)
%  input: confusion matrix where each row corresponds to the ground truth and the colomns are the predicted ones
% Outputs
%  m: F-measure for each class
%  normM: weighted F-measure usable for multi-class
if nargin<3
    mode = 'full';
end
        nClass = size(input,1);
switch method
    case 'F'
        %norma = zeros(1,nClass);
        m = zeros(1,nClass);
        for iClass = 1:nClass
            if strcmp(mode,'full')
            TP = input(iClass,iClass);
            %d = diag(input);
            %z = 1:nClass;
            %TN = sum(d(z~=iClass));
            FP = sum(input(:,iClass)) -TP;
            FN = sum(input(iClass,:))-TP;
            sen = TP/(TP+FN);
            prec = TP/(TP+FP);
            m(iClass) = 2*sen*prec/(sen+prec);    
            m(isnan(m)) = 0;
            s = input(1:end,:);
            normaWithnull(iClass) = m(iClass)*sum(input(iClass,:))/sum(s(:));
            
            s = input(2:end,:);
            normaWithoutNull(iClass) = m(iClass)*sum(input(iClass,:))/sum(s(:));
            else
                input = input./repmat(sum(input,2),[1,2]); %GOODL
                
                TN = input(1,1);
                TP = input(2,2);
                FP = input(1,2);
                FN = input(2,1);
                prec = TP/(TP+FP);
                recall = TP/(TP+FN);
                f = 2*prec*recall/(prec+recall);
                if isnan(f),f = 0;end;                    
                return;
            end
        end
        m(isnan(m)) = 0;
        normaWithnull(isnan(normaWithnull)) = 0;
        normaWithoutNull(isnan(normaWithoutNull)) = 0;
        %m = norma;
        normMact = sum(normaWithoutNull(2:end));
        normMnull = sum(normaWithnull(1:end));
        f.normaWithnull=normaWithnull;
        f.normaWithoutnull = normaWithoutNull(2:end);
        %normM = mean(m(2:end));
    case 'A',
        %input = input./repmat(sum(input,2),[1,nClass]);
        a = 1:size(input,1);
        dd = input(a,a);
        f.avgDiag = sum(diag(dd))/sum(dd(:));
end

res = resultmean;
[nSubject nClassifier] = size(res.acc);
for iSubject=1:nSubject
    for iClassifier = 1:nClassifier
        for mode = 1:2
            if ~isfield(res.acc{iSubject,iClassifier}(mode),'ROC')
                continue;
            end
            nAct = length(res.acc{iSubject,iClassifier}(mode).ROC);
            tt = resultmean.acc{iSubject,iClassifier}(1,mode).ward.t.Conf(2:end,:);
            %%tt = res.acc{iSubject,iClassifier}(mode).conf(2:end,:);
            allSamples = sum(tt(:));
            weightedROCtemp = 0;
            for iAct = 1:nAct
                rocFNr = [0 res.acc{iSubject,iClassifier}(mode).ROC{iAct}.FNr 1];
                rocTPr = [0 res.acc{iSubject,iClassifier}(mode).ROC{iAct}.TPr 1];
                nt = length(rocFNr);
                s = 0;
                for it = 2:nt
                    s = s+(rocTPr(it)+rocTPr(it-1))/2*(rocFNr(it)-rocFNr(it-1));
                end  
                aucurve{iSubject,iClassifier}{mode}(iAct) = s;
                
                %%nActSample = sum(res.acc{iSubject,iClassifier}(mode).conf(iAct+1,:));%skiping null
                nActSample = sum(resultmean.acc{iSubject,iClassifier}(1,mode).ward.t.Conf(iAct+1,:));
                w = nActSample/allSamples;
                weightedROCtemp = weightedROCtemp+w*s;
            end
            weightedROC{iSubject,iClassifier}(mode)=weightedROCtemp;
        end
    end
end
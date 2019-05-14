function feature=featureExtraction(data,param,type)
%
%Input:
% data:nxm, n:#of patterns, m: number of features
% method:'raw','mean','meanVar'
% window: window length
% type: 1 --- for movtimavg
%       2 --- for windowingLabels

switch param.method
    case 'raw', feature = data;
        
    case 'mean'
        if (type==1)
        feature = movtimavg(data,param.window,param.step);
        end
        if (type==2)
        feature = windowingLabels(data,param.window,param.step);
        end
        
    case 'meanVar',feature = movtimavg(data,param.window,param.step,1);
        if (type==1)
        feature = movtimavg(data,param.window,param.step,1);
        end
        if (type==2)
        feature = windowingLabels(data,param.window,param.step);
        end
end
function [r, lab, meanLenOU] = mset_segments( pred, ground, varargin )
% [r, lab] = mset_segments( pred, ground, varargin )
%
%  Categorises each contiguous segment 
%  written by Jamie Ward, 2005
%
% input: 
% pred, ground = [start_time stop_time label]
%             with all labels mapped contiguously onto
%             0*,1,2,3,..
%
% Any times which are not explicitly covered by a label (e.g. between
% segments) is regarded as NULL. This is automatically assigned the label 0
% * You can specify NULL explicity by using the label 0. 
% If, however, you wish to use label 1 for NULL then set
% varargin='NULL_ONE'
%
%   ( Bug/Feature: the evaluation covers time 0 until the last stop_time in
%   the ground truth.)
%%%
%
% output: [r, lab] 
%
% r=[seg_start_time seg_end_time error_code] == label to segment mapping 
%  where error_code corresponds to the following error pairings:
%   0 = Match
%  18 = ID
%  20 = IU
%  24 = IF
%  34 = OD
%  36 = OU
%  66 = MD
%
% lab = [pred_label ground_label] for each segment in r
%

NLL=0; inNLL=0;
if ~isempty(varargin)
    for n=1:length(varargin)
        if findstr(varargin{n},'NULL_ONE')                
            NLL=1;
            inNLL=1; % NULL is already 1
        end
    end
end

if NLL==0  
    % Shift all labels so that NULL (assumed to be 0) is moved to '1'
    if ~isempty(pred)
        pred(:,3)=pred(:,3)+1;
    end
    ground(:,3)=ground(:,3)+1;
    inNLL=0; NLL=1; 
end


MATCH = 0; 
DEL = 2^1; % 2
UNDER = 2^2; % 4
FRAG = 2^3; % 8
INS = 2^4; % 16
OVER = 2^5; % 32
MERGE = 2^6; % 64



% Divide into segments
% 1. merge pred and ground into a single sorted column of event stop times 
% but preserve combination of associated labels.

% Ground
% Make segment times into a single column, i.e.[start1 stop1 start2 stop2]'
gndT=ground(:,1:2)';
gndT=gndT(:);
gndT(:,2)=ones(length(gndT),1);
gndT(2:2:end,2) = ground(:,3) ; % add the labels - mark label corresponding to stop time!

% Remove zero length segments 
nonzero = [ 1 ; find( 0~= gndT(2:end,1) - gndT(1:end-1,1) ) + 1];
gndT=gndT(nonzero,:);
nGndT=size(gndT,1);

% Merge consecutive segments where there is no change
keep=[1];
for i=2:nGndT-1
    now = gndT(i,2:end);
    next = gndT(i+1,2:end);    
    if now==next
        % remove now.
    else
        keep = [keep; i];
    end
end
keep=[keep; nGndT]; 
gndT=gndT(keep,:);
nGndT=size(gndT,1);

% Predictions
if isempty(pred)
    predT(1,1:2) = [0 NLL];
else
    predT=pred(:,1:2)';
    predT=predT(:);
    predT(:,2)=ones(length(predT),1);
    predT(2:2:end,2) = pred(:,3) ; % add the labels
end

% Remove zero length segments 
nonzero = [ 1 ; find( 0~= predT(2:end,1) - predT(1:end-1,1) ) + 1];
predT=predT(nonzero,:);
nPredT = size(predT,1);

% Merge consecutive segments where there is no change
keep=[1];
for i=2:nPredT-1
    now = predT(i,2:end);
    next = predT(i+1,2:end);    
    if now==next
        % remove now.
    else
        keep = [keep; i];
    end
end
keep=[keep; nPredT]; 
predT=predT(keep,:);
nPredT=size(predT,1);


% Ensure prediction and ground have the same length
if gndT(end,1) > predT(end,1)
    predT = [predT; [gndT(end,1) NLL]];
elseif gndT(end,1) < predT(end,1)
    %! gndT = [gndT; [predT(end,1) NLL]];
    %! Truncate predictions
    i=length(predT)-1;
    while gndT(end,1) < predT(i,1),
        i=i-1;
    end
    if predT(i,1)==gndT(end,1)
        predT = predT(1:i,:);
    else       
        % Truncate an event
        predT = predT(1:i+1,:);
        predT(i+1,1) = gndT(end,1);
    end
else
    % Ok.
end

[nGnd, col] = size( gndT );
[nPred, col] = size( predT );

% Change the label columns to represent [stopTime groundL predL]
gndT(:,3) = (-1)*ones(nGnd,1);
predT(:,3) = predT(:,2);
predT(:,2) = (-1)*ones(nPred,1);

% Now merge both.
segT = [gndT;predT];
[segT(:,1), iSegT] = sort(segT(:,1)); % sort only according to timing
segT(:,2:3) = segT(iSegT, 2:3); % Re assign labels
nSegT = size(segT,1);

% The beginning time is always one (with class null)
segT = [0 NLL NLL; segT];

% spread the labelling where necessary - spread backwards from marked
% .. because a label is first mentioned at the _end_ of its segment.
for idx= nSegT:-1:1
   gL = segT(idx+1,2); % Current gnd label
   pL = segT(idx+1,3); % Current pred label    
   mergeSeg = (segT(idx+1,1)-segT(idx,1) == 0); % Zero length segment
   
   if mergeSeg==true
       % Zero length segment, so merge both entries
       % to swamp out any '-1's
       if segT(idx,2) > -1
           gL=segT(idx,2);
       end
       if segT(idx,3) > -1
           pL=segT(idx,3);
       end
      segT(idx:idx+1, 2) = gL;
      segT(idx:idx+1, 3) = pL;
   end   
   
   % Usually at the end of the segment file, no more information: set to
   % null.
   if gL == -1
      segT(idx+1,2) = NLL;
   end
   if pL == -1
      segT(idx+1,3) = NLL;
   end
   
   % Now set preceeding labels
   if (segT(idx,2) == -1) 
       segT(idx,2) = gL; 
   end
   
   if (segT(idx,3) == -1) 
       segT(idx,3) = pL;
   end
      
end

% Remove zero length segments 
nonzero = [ 1 ; find( 0~= segT(2:end,1) - segT(1:end-1,1) ) + 1];
segT=segT(nonzero,:);

nSegT=size(segT,1); % Re-evaluate length

% Merge consecutive segments where there is no change
keep=[1];
for i=2:nSegT-1
    now = segT(i,[2 3]);
    next = segT(i+1,[2 3]);    
    if now==next
        % remove now.
    else
        keep = [keep; i];
    end
end
keep=[keep; nSegT];
segT=segT(keep,:);

nSegT=size(segT,1); % Re-evaluate length
segEtype = zeros(nSegT,1);

% Go through all the ground truth events
start = 0;
for i = 1:nGnd;
   stop = gndT(i);
   % Find index of all segments within this event
   isegs = find( start < segT(:,1) & segT(:,1) <= stop );
   % Find index of all matching segments
   isegsmatch = isegs( segT(isegs,2)==segT(isegs,3) );
   
   if isempty(isegsmatch)
     % All segments within this event are deletions
     segEtype(isegs) = segEtype(isegs) + DEL;     
   else
     % Any segments occurring before the first or after the last match is an underfill
     iunderfillsegs = [isegs(isegs < isegsmatch(1)); isegs(isegs > isegsmatch(end))];
     if ~isempty(iunderfillsegs)
         segEtype(iunderfillsegs) = segEtype(iunderfillsegs) + UNDER;     
     end
     
     % Any segments occurring between correctly matched segments of the
     % same event is a fragmenting error
     inonfragsegs = [isegsmatch; iunderfillsegs];
     for n=1:length(isegs)
         if isempty( find(isegs(n)==inonfragsegs) );
             segEtype(isegs(n)) = segEtype(isegs(n)) + FRAG;
         end
     end

   end
   start = stop;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Go through all the prediction events
start = 0;
for i = 1:nPred;
   stop = predT(i); 
   % Find index of all segments within this event
   isegs = find( start < segT(:,1) & segT(:,1) <= stop );
   % Find index of all matching segments
   isegsmatch = isegs( segT(isegs,2)==segT(isegs,3) );
   
   if isempty(isegsmatch)
     % All segments within this event are insertions
     segEtype(isegs) = segEtype(isegs) + INS;     
   else
     % Any segments occurring before the first or after the last match is
     % an overfill
     iunderfillsegs = [isegs(isegs < isegsmatch(1)); isegs(isegs > isegsmatch(end))];
     if ~isempty(iunderfillsegs)
         segEtype(iunderfillsegs) = segEtype(iunderfillsegs) + OVER;     
     end
     
     % Any segments occurring between correctly matched segments of the
     % same event is a merger error
     inonfragsegs = [isegsmatch; iunderfillsegs];
     for n=1:length(isegs)
         if isempty( find(isegs(n)==inonfragsegs) );
             segEtype(isegs(n)) = segEtype(isegs(n)) + MERGE;
         end
     end

   end
   start = stop;
end


if inNLL==0
   % Shift all labels down again
   segT(:,[2 3])=segT(:,[2 3])-1;
end

%%%%%
% Convert back into standard segment format
startT = segT(1:end-1,1);
% remove the superfelous first row from 'stopT'
segmentResults = [startT segT(2:end,:) segEtype(2:end,:)];
r=segmentResults(:,[1 2 5]);
% corresponding ground and prediction labels
lab=segmentResults(:,[3 4]);

% finding length of overfill or underfill
iOU=segmentResults(:,5)==36;
segOU=segmentResults(iOU,1:2);
meanLenOU=mean(segOU(:,2)-segOU(:,1));
end

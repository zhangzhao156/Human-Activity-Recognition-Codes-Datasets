function [t, s, e, meanLenOU] = mset( pred, ground, varargin )
% [t, s, e] = mset(pred, ground, varargin)
%
% continuous context evaluation method which
% fills out a Multiclass Segment Error Table (SET)
% written by Jamie Ward, 2005
%
% input: 
% pred, ground = [start_time stop_time label]
%             with all labels mapped contiguously onto
%             0*,1,2,3,..
%
% Any times which are not explicitly covered by a label (e.g. between
% segments) is regarded as NULL. This is automatically assigned the label 0
% * You can specify NULL explicity by using the label 0. 
% If, however, you wish to use label 1 for NULL then set varargin='NULL_ONE'
%
%   ( Bug/Feature: the evaluation covers time 0 until the last stop_time in
%   the ground truth.)
%
% varargin =>
% The default number of classes is taken from the ground and prediction information, however if
% these inputs do not contain representatives of all classes, then you can
% specify, for example: 'nClasses', number  (note: includes NULL class)
%
%
% output: 
% t== time based result 
% s== segment result
%   .T = Total
%   .ID = Insertion Deletion, etc..
%   .IU
%   .IF
%   .OD
%   .OU
%   .MD
%   .Conf - regular confusion matrix
%
% e==event based result
% .T = total
% .I, .D, .M, .F 
% (e. also has event timing errors) 
% .Pre = preemption (overfill)
% .Pro = prolongation (overfill)
% .Short = shortening (underfill)
% .Delay = delay (underfill)
%
%
%%%%%%%%%%%%%%%
% Uses: mset_segments.m
%

NLL=0;
nClasses=0;

if ~isempty(varargin)
    for n=1:length(varargin)
        if findstr(varargin{n},'NULL_ONE')                
            NLL=1;
        end
        if findstr(varargin{n},'nClasses')
            if length(varargin)==n
                error('nClasses needs an argument');
            end
            nClasses=varargin{n+1};
        end
    end
end

if NLL==0  
    % Shift all labels so that NULL (assumed to be 0) is moved to '1'
    if ~isempty(pred)
        pred(:,3)=pred(:,3)+1;
    end
    ground(:,3)=ground(:,3)+1;
    NLL=1; 
end

nClasses = max( [nClasses; max(ground(:,3))] );

% initialise the outputs
t.Conf = zeros(nClasses,nClasses);
t.ID=t.Conf; t.IU=t.Conf; t.IF=t.Conf; t.OD=t.Conf; t.OU=t.Conf; t.MD=t.Conf;
s.Conf=t.Conf;
s.ID=s.Conf; s.IU=s.Conf; s.IF=s.Conf; s.OD=s.Conf; s.OU=s.Conf; s.MD=s.Conf;

%e.Conf = t.Conf; e.ID=t.Conf; e.IU=t.Conf; e.IF=t.Conf; e.OD=t.Conf; e.MD=t.Conf;
e.T = zeros(nClasses,1); e.I = e.T; e.D = e.T; e.M = e.T; e.F = e.T; e.Corr=e.T;
e.Short = e.T; e.Delay = e.T; e.Pre = e.T; e.Pro = e.T;

% error codes from mset_segments
MATCH = 0;
DEL = 2^1; % 2
UNDER = 2^2; % 4
FRAG = 2^3; % 8
INS = 2^4; % 16
OVER = 2^5; % 32
MERGE = 2^6; % 64
ID = INS+DEL; %18;
IU = INS+UNDER; %20;
IF = INS+FRAG; %24;
OD = OVER+DEL; %34;
OU = OVER+UNDER; %36;
MD = MERGE+DEL; % 66;

%[r lab] = mset_segments( pred, ground, 'nClasses',nClasses,'NULL',NLL);
[r lab meanLenOU] = mset_segments( pred, ground, 'NULL_ONE' );

segD = r(:,2)-r(:,1);

for n=1:length(segD)
    % read each segment info into the appropriate table
    grL = lab(n,1); prL = lab(n,2);
    switch(r(n,3))
        case ID
            t.ID(prL,grL) = t.ID(prL,grL) + segD(n);
            s.ID(prL,grL) = s.ID(prL,grL) + 1;
        case IU
            t.IU(prL,grL) = t.IU(prL,grL) + segD(n);
            s.IU(prL,grL) = s.IU(prL,grL) + 1;
        case IF
            t.IF(prL,grL) = t.IF(prL,grL) + segD(n);
            s.IF(prL,grL) = s.IF(prL,grL) + 1;
        case OD
            t.OD(prL,grL) = t.OD(prL,grL) + segD(n);
            s.OD(prL,grL) = s.OD(prL,grL) + 1;
        case OU
            t.OU(prL,grL) = t.OU(prL,grL) + segD(n);
            s.OU(prL,grL) = s.OU(prL,grL) + 1;
        case MD
            t.MD(prL,grL) = t.MD(prL,grL) + segD(n);
            s.MD(prL,grL) = s.MD(prL,grL) + 1;
        otherwise
            % Match
            %disp(strcat(num2str(r(n,3)), num2str(prL),num2str(grL)));
    end

    % Fill out regular confusion matrix
    t.Conf(prL,grL) = t.Conf(prL,grL) + segD(n);
    s.Conf(prL,grL) = s.Conf(prL,grL) + 1;
end

% Overall sum
t.T = sum(t.Conf);
s.T = sum(s.Conf);



%%% Counting events

% Make segment times into a single column, i.e.[start1 stop1 start2 stop2]'
gndT=ground(:,1:2)';
gndT=gndT(:);
gndT(:,2)=ones(length(gndT),1);
gndT(2:2:end,2) = ground(:,3) ; % add the labels - mark label corresponding to stop time!

if isempty(pred)
    predT(1,1:2) = [0 NLL];
else
    predT=pred(:,1:2)';
    predT=predT(:);
    predT(:,2)=ones(length(predT),1);
    predT(2:2:end,2) = pred(:,3) ; % add the labels
end


% Ensure prediction and ground have the same length
if gndT(end,1) > predT(end,1)
    predT = [predT; [gndT(end,1) NLL]];
elseif gndT(end,1) < predT(end,1)
    gndT = [gndT; [predT(end,1) NLL]];
else
    % Ok.
end

[nGnd, col] = size( gndT );
[nPred, col] = size( predT );

% Go through all the ground truth events
start=0;
for i = 1:nGnd;
    stop = gndT(i);

    if stop-start > 0
        % Find index of all segments within this event
        isegs = find( start < r(:,2) & r(:,2) <= stop );
        numIsegs = length(isegs);

        % Deleted; or fragmented - how many times?
        del = 0; frag = 0;
        for n=1:numIsegs
            grL = lab(isegs(n),1);
            switch(r(isegs(n),3))
                case {ID,MD,OD}
                    del = del + 1;
                case IF
                    frag = frag + 1;
                otherwise
                    % Correct..
            end
        end

        % Negative timing error (Underfill)?
        delay = 0; shortening = 0;
        if del==0
            for n=1:numIsegs
                switch(r(isegs(n),3))
                    case {IU,OU}
                        if n==1 % at beginning
                            delay = delay+1;
                        elseif n==numIsegs % at end
                            shortening = shortening+1;
                        else end
                    otherwise
                        % No negative timing error
                end
            end
        end

        e.T(grL) = e.T(grL) + 1;

        if del>0
            e.D(grL) = e.D(grL) + 1;
        elseif frag>0
            e.F(grL) = e.F(grL) + 1;%frag;
        else
            e.Corr(grL) = e.Corr(grL) + 1;
        end

        if delay > 0
            e.Delay(grL) = e.Delay(grL) + 1;
        end
        if shortening > 0
            e.Short(grL)= e.Short(grL) + 1;
        end

    end

    start=stop;
end


% Go through all the prediction events
start=0;
for i = 1:nPred;
    stop = predT(i);
    if stop-start > 0
        % Find index of all segments within this event
        isegs = find( start < r(:,2) & r(:,2) <= stop );
        numIsegs = length(isegs);
        ins=0;merg=0;
        % Insertion; or merge - how many?
        for n=1:numIsegs
            prL = lab(isegs(n),2);
            switch(r(isegs(n),3))
                case {ID,IU,IF}
                    ins = ins + 1;
                case MD
                    merg = merg + 1;
                otherwise
                    % Correct.
            end
        end

        % Positive timing error (Overfill)?
        pre = 0; pro = 0;
        if ins==0
            for n=1:numIsegs
                switch(r(isegs(n),3))
                    case {IU,OU}
                        if n==1 % at beginning
                            pre = pre+1;
                        elseif n==numIsegs % at end
                            pro = pro+1;
                        else end
                    otherwise
                        % No positive timing error
                end
            end
        end


        if ins>0
            e.I(prL) = e.I(prL) + 1;
        elseif merg>0
            e.M(prL) = e.M(prL) + 1;% merg;
        end

        if pre > 0
            e.Pre(prL) = e.Pre(prL) + 1;
        end

        if pro > 0
            e.Pro(prL)= e.Pro(prL) + 1;
        end

    end

    start=stop;
end


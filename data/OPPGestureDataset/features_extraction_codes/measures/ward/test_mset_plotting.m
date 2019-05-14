
% Example of how to use mset scripts, Jamie Ward, 2006
%
% - copy texoutT and texoutE into a .tex file for tabulated results.
%
% eval = list of events to be evaluated, [start_time stop_time label]
% ground = ground truth events, [start_time stop_time label]
% nClasses = number of classes
%
% Note: class labels should be 1, 2, 3,.. (we want to build an nClasses x
% nClasses matrix) NULL=0 is implicit: 
%

% ground and eval format:  [start_time stop_time label]
            
ground    = [ 10      25     1 ...
            ; 30      40     2 ...
            ; 40      42     3 ...
            ; 50      67     2 ...
            ; 69      70     2 ...
            ; 75      80     1 ...
            ; 90      92     1 ];

        % Some errors thrown into the evaluated events
eval    =   [ 12    22     1 ... 
            ; 32    35     2 ...  
            ; 37    45     3 ...              
            ; 54    56     2 ...
            ; 60    65     1 ... 
            ; 66    72     2 ...
            ; 75    80     2 ...
            ; 82    84     3 ...
            ];

            
 CLASSES = {'NULL';'1';'2';'3'};           
 nClasses = size(CLASSES,1); % including NULL class
 
% In this example, we treat NULL as a special case
[t,s,e] = mset(eval, ground, 'nClasses', nClasses );

% Print out the SET tables with segment and timing information
texoutT = mset_print(nClasses, 'seg',s, 'time', t,'groups',{[2:nClasses] [1]},'labels',{'Pos','N'});
% Print out the event errors
texoutE = mset_print(nClasses, 'eventTime',e );


% We can access the segment by segment info directly using
[r l] = mset_segments( eval, ground );

% Plot the segment error results besides the predictions and ground truth
fs = 1; 
groundplot = read_seg_info( ground, fs, [], -1, 'lab' );
nLen = length(groundplot);
evalplot = read_seg_info( eval, fs, [], nLen, 'lab' );


segout = read_seg_info( r, fs, [], -1, 'lab' );
% The error codes from r need converting first...                         
segout(segout==18) = 1;  % ID
segout(segout==20) = 2;  % IU
segout(segout==24) = 3;  % IF
segout(segout==34) = 4;  % OD
segout(segout==36) = 5;  % OU
segout(segout==66) = 6;  % MD


ax(1)=subplot(2,1,2);plot(segout,'r+');     
    set(ax(1),'YTick',0:7); 
    set(ax(1),'YTickLabel',{'Match','ID','IU','IF','OD','OU','MD'});
    set(ax(1),'YLim',[0 6]);
ax(2)=subplot(2,1,1);
      plot(groundplot,'g^'); hold on;
      plot(evalplot+0.2,'b.'); 
      legend({'eval','ground'});
      title('SET evaluation');
      set(ax(2),'YTick', 0:nClasses );
      set(ax(2),'yTickLabel',{CLASSES{:},' '});
linkaxes(ax,'x'); 

% Plot the bar charts now
plot_mset_errors(2,'piefull','test',t,'test',s);
figure;
plot_mset_errors(2,'barfull','test',t,'test',s);





function segment_info = write_seg_info( sample_rate, input, format );
% segment_info - minimum 3 column segment info format 
% sample_rate - refers to the rate of the input, i.e. if seg format = 1
% input depends on format:
% format string indicates what input should look like, e.g. 'lab' or 'seg'
% e.g. in=label_array, format = 'lab'  or in=seg_info, format='seg'
%Input format for 'seg'
% [Start_time Stop_time Label ..Labeln ] (times given according to sample_rate)
%Input format for 'lab' 
% [Label1, ..Labeln] (continuous at sample_rate)
%
%N.B. any label specified as 0 is not given a segment allocation.
% Segments are marked in times from zero.

% Output format
SEG_FILE_FS = 1; % Seconds.

START = 1;
STOP = 2;

if isempty(input)
    segment_info = [];
    return;
end

% input format..
switch format
case 'lab'
    
    % Find all the segment locations, start and stop times
    lab_changes = find(diff( input ));
    
    if input(1)==0
        % First sample is part of a null segment.
        seg_start = [lab_changes]; 
        seg_stop = [ lab_changes(2:end); length(input) ]; %% add the final section, ending at len(input)
    else
        % First sample is part of an actual segment - it therefore begins
        % at sample=1 -> time=0.
        seg_start = [0; lab_changes]; %%  add initial section, starting at sample 1, time zero.
        seg_stop = [ lab_changes; length(input) ]; %% add the final section, ending at len(input)
    end
    
%    % Keep only the non-zero labelled segments - looking only at the first
%    % (main) label column.
%    non_zero_positions = find( input( seg_start, 1 ) ~= 0 ) + 1;
%    seg_start = seg_start( non_zero_positions );
%    seg_stop = seg_stop( non_zero_positions );
    
    % How many label columns are given
    num_labelCols = size( input, 2 ) ;
    segfile_len = length(seg_start);

    segment_info = zeros( segfile_len, num_labelCols+2 );
    
    % Fill out the labels corresponding to these times
    for i=1:num_labelCols
        segment_info(:,STOP+i) = input(seg_stop,i);
    end

case 'seg'
    % We write these segment files out as specified - no removing zero
    % labels.
    
    seg_start = input(:,1);
    seg_stop = input(:,2);    

    % How many label columns are given
    num_labelCols = size( input, 2 ) - 2;

    segment_info = zeros( size(input) );
    
    for i=1:num_labelCols
        segment_info(:,STOP+i) = input(:,2+i);
    end
        
otherwise
    disp('not a valid format');
    return;
end

if length( seg_start ) > 0
    segment_info(:,START) = seg_start;
    segment_info(:,STOP) = seg_stop;    
end

% Remove the labelled segments with a 0 first label..
first_labels = segment_info(:,STOP+1);    
segment_info = segment_info(first_labels~=0, :);


% Convert from the input times at sample_rate to the output in seconds
segment_info(:,[START,STOP]) = segment_info(:,[START,STOP]) * (SEG_FILE_FS / sample_rate) ;


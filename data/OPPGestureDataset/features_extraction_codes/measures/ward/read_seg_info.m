    
function out = read_seg_info( segment_info, sample_rate, include_classes, seq_len, format );
%function out = read_seg_info( segment_info, sample_rate, include_classes,
%seq_len, format );
% out = read_seg_info( segment_info, sample_rate, include_classes, seq_len, format );
% segment_info - 3 column segment information [start stop label] (all times in
% seconds,label in ascii)
% sample_rate - refers to the rate of the output
% format string indicates what output should look like, e.g. 'lab' or 'seg'
% e.g. 
%Output format for 'seg'
% [Start_time Stop_time Label ..Labeln ] (times given according to sample_rate)
%Output format for 'lab' 
% [Label1, ..Labeln] (continuous at sample_rate)\
% if seq_len = -1, then the label array is created until the last marked segment info.
%
% Label output is zero if it is of an unknown class or it is between segments (i.e. null)
% Segments are marked in times from zero.

% Input format
SEG_FILE_FS = 1; % Seconds.

START = 1;
STOP = 2;
if isempty(segment_info)
   switch format
    case 'lab' 
        if seq_len==-1
            out = 0;
        else
            out = zeros(seq_len,1);
        end
    case 'seg' 
        out = [];
    otherwise
        out=0;
    end
    return;
end

if iscell(include_classes)
    include = char(include_classes{:,2});
else
    include = char(include_classes);
end

% Convert from the input times to  sample_rate of output
segment_info(:,[START,STOP]) = segment_info(:,[START,STOP]) * (sample_rate / SEG_FILE_FS) ;

num_labels = size( segment_info, 2 ) - 2;
%%% Check if START or STOP changes..!
label_positions = STOP + [1:num_labels];

switch format
case 'lab'

    % Ensure integer values
    segment_info(:,[START,STOP]) = floor( segment_info(:,[START,STOP]) );
    % Ensure that no segment starts at zero..
    start_pos = segment_info(1,START);
    
        %shift up all segment starts and stops by 1--it cannot be zero
        %indexed for now.
        segment_info(:,START:STOP) = segment_info(:,START:STOP) + 1;
    
        last_seg = length( segment_info(:,2) );
        
        if (seq_len==-1) 
            seq_len = ceil( segment_info(last_seg,STOP) );
        end
        
        % Find the longest possible length of the sequence
        longest_len = ceil(max( seq_len, segment_info(last_seg,2) ));
        sLabels=zeros( longest_len, num_labels );
        
        % Already marked.
        for seg=1:last_seg
            
            start_pos = floor( segment_info(seg,START) );

            end_pos = ceil( segment_info(seg,STOP) );
            
            % length inlcudes start and stop elements.
            len= end_pos-start_pos;%+1 ;
            
            % Create a continuous label array.
            L = ones( len, 1 ) * segment_info(seg, label_positions);
            % L = ones( len, 1:num_labels ) * segment_info(seg,label_positions); Sept 2007
            sLabels( start_pos : end_pos-1, 1:num_labels ) = L;         
       end
                
        % Truncate length according to parameters
        Labels= sLabels( 1:seq_len, : );

        if ~isempty( include_classes )
            % zero those classes we do not work with 
            L = zeros(size(Labels));
            for i=1:length(include)
                L( Labels == include(i) ) = include(i);
            end
            Labels = L;
        end


    out = Labels;    
        
case 'seg'
    
    out = segment_info(:, [START, STOP, label_positions] );        
    if ~isempty( include_classes )       
        % Find positions of known labels
        iSegments =[];
        
        for i=1:length(include)
           iSegments = [ iSegments; find(out(:,3)==include(i)) ];
        end
        % Re-order the segment indexes
        iSegments=sort(iSegments);
        % remove classes we do not work with
        out = out( iSegments, : );
        
    else
        % accept any labelling.
        out = segment_info(:, [START, STOP, label_positions] );    
    end

    if (seq_len~=-1) 
        % Truncate those segment entires that go beyond the specified data
        % length
        out = out( (out(:,STOP) < seq_len), : );
    end
    
    
otherwise
    disp('not a valid format');
    return;
end


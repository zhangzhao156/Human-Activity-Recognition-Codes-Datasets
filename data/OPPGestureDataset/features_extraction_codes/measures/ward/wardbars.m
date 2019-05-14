function [handle,output]=wardbars(label,data)
% This function is a modification of the function plot_mset_errors
% developed by Jamie Ward et al.
%--------------------------------------------------------------------------
% The inputs are: labels and the data for which the accuracies have to be
% calculated
% The outputs are:
%   handle: handle for the figure plotted by the function
%   output: an array of various accuracy measures

nSets=length(label);
nSubj=size(data,1);
nClassifiers=size(data,2);
handle=figure;
distBar=0.25;   %parameter for the distance between the bars in the plots
m=ceil((nClassifiers/2)*distBar);
for iSubj=1:nSubj
    for iSet=1:nSets % Gather all the data
        L(iSubj)=m;
        tm=data{iSubj,iSet};
        % Sum up times over all files
        nFiles = length(tm);
        tID=0;tOD=0;tMD=0;tIU=0;tOU=0;tIF=0;tConf=0;tT=0;
        for i=1:nFiles
            tOD = tOD+tm(i).OD;
            tID = tID+tm(i).ID;
            tMD = tMD+tm(i).MD;
            tIU = tIU+tm(i).IU;
            tOU = tOU+tm(i).OU;
            tIF = tIF+tm(i).IF;
            tConf = tConf + tm(i).Conf;
            tT = tT + tm(i).T;
        end
        
        nClasses=length(tT);
        Ngrp=[1]; Pgrp=[2:nClasses];
        
        % w.r.t. Null
        Del(iSet) = sum( [sum(tID(Ngrp,Pgrp)) sum(tOD(Ngrp,Pgrp)) sum(tMD(Ngrp,Pgrp))] );
        Und(iSet) = sum( [sum(tIU(Ngrp,Pgrp)) sum(tOU(Ngrp,Pgrp))] );
        Fra(iSet) = sum(  sum(tIF(Ngrp,Pgrp)) );
        Ins(iSet) = sum( [sum(sum(tID(Pgrp,Ngrp))) sum(sum(tIU(Pgrp,Ngrp))) sum(sum(tIF(Pgrp,Ngrp)))] );
        Ove(iSet) = sum( [sum(sum(tOD(Pgrp,Ngrp))) sum(sum(tOU(Pgrp,Ngrp)))] );
        Mer(iSet) = sum(  sum(sum(tMD(Pgrp,Ngrp))) );
        % Substitutions
        sID(iSet) = sum(sum(tID(Pgrp,Pgrp)));
        sIU(iSet) = sum(sum(tIU(Pgrp,Pgrp)));
        sIF(iSet) = sum(sum(tIF(Pgrp,Pgrp)));
        sOD(iSet) = sum(sum(tOD(Pgrp,Pgrp)));
        sOU(iSet) = sum(sum(tOU(Pgrp,Pgrp)));
        sMD(iSet) = sum(sum(tMD(Pgrp,Pgrp)));
        % Correct
        I = eye(nClasses,nClasses);%identity matrix
        C = sum(tConf() .* I);%sum of diagonal of confusion martix
        
        pC(iSet) = sum(C(Pgrp));
        nC(iSet) = sum(C(Ngrp));
        T(iSet) = sum(tT);
        Pos(iSet)= sum(tT(Pgrp));
        Neg(iSet)= sum(tT(Ngrp));
        
        sub(iSet)= sum(sum(tConf(Pgrp,Pgrp))-C(Pgrp));
        predPos(iSet)= sum(sum(tConf(Pgrp,:)));
        predNeg(iSet)= sum(sum(tConf(Ngrp,:)));
        
    end
    
    subst= sID+sIU+sIF+sOD+sOU+sMD;
    
    barlegend= {'TP';'TN';'Overfill';'Underfill';'Merge';'Insertion';'Fragmentation';'Deletion';'Substitution'};
    barlegendshort = {'S';'D';'F';'I';'M';'U';'O';'TN';'TP'};
    explode =[1 1 1 1 1 0 0 0 0];
    errorcolors=[1 0.3 0; 1 0.5 0; 1 0.7 0; 1 0.9 0; 1 1 0 ; 1 1 0.4; 1 1 0.6; 1 1 0.8; 1 1 1];
    
%         for i=1:nSets
%             labels{i} = {...
%             sprintf('Subst [%2.0f] %2.1f%%  ',subst(i),100*subst(i)/T(i))...
%             ,sprintf('D [%2.0f] %2.1f%%  ',Del(i),100*Del(i)/T(i))...
%             ,sprintf('F [%2.0f] %2.1f%%  ',Fra(i),100*Fra(i)/T(i))...
%             ,sprintf('I [%2.0f] %2.1f%%  ',Ins(i),100*Ins(i)/T(i))...
%             ,sprintf('M [%2.0f]\n %2.1f%% ',Mer(i),100*Mer(i)/T(i))...
%             ,sprintf('U [%2.0f]\n %2.1f%% ',Und(i),100*Und(i)/T(i))...
%             ,sprintf('O [%2.0f]\n %2.1f%% ',Ove(i),100*Ove(i)/T(i))...
%             ,sprintf('CorrNull \n%2.1f%% ',100*nC(i)/T(i))...
%             ,sprintf('CorrPos \n%2.1f%% ',100*pC(i)/T(i))...
%             };
%         end
    
    % columns of M is stacked into a bar
    % there are as many bars as the rows of M
    M(:,9)= pC;
    M(:,8)= nC;
    M(:,7)= Ove;
    M(:,6)= Und;
    M(:,5)= Mer;
    M(:,4) = Ins;
    M(:,3)= Fra;
    M(:,2) = Del;
    M(:,1)= subst;
    Mscaled=bsxfun(@rdivide,M,sum(M,2))*100;
    SEL=5;
    
    % plotting the bars
    placebars=zeros(nClassifiers,1);
    if(rem(nClassifiers,2)==1)
        for i=1:nClassifiers
            placebars(i,1)=m+(i-ceil(nClassifiers/2))*distBar;
        end
    else
        for i=1:nClassifiers
            if i<=nClassifiers/2
                placebars(i,1)=m+(i-(nClassifiers/2)-1)*distBar;
            else
                placebars(i,1)=m+(i-(nClassifiers/2))*distBar;
            end
        end
    end
    bar(placebars,Mscaled,.8,'stacked'), colormap(errorcolors);
    set(gca,'nextplot','add');
    
        
    m=m+ceil(nClassifiers*distBar)+1;

%%
%%%The following part calculates Generality, Null activity,precisoin etc %%
%   It can be put in a table
%
     [nMeth,nErr] = size(M);
     for meth = 1:nMeth

        Generality = 100*Pos(meth)/T(meth);
        output(iSubj,meth).NULLactivity=(100-Generality);
        %infoLabel = sprintf( 'NULL activity = %3.1f%% (%.1f / %.1f) \n', 100-Generality, Neg(meth), T(meth) );

        output(iSubj,meth).recall = sum(M(meth,end))*100./Pos(meth);
        output(iSubj,meth).precision = sum(M(meth,end))*100./predPos(meth);
        output(iSubj,meth).fp = (Neg(meth)-sum(M(meth,end-1)))*100./Neg(meth);
        %restxt = sprintf('Precision: %2.1f%%, Recall: %2.1f%%, FP: %2.1f' , precision, recall, fp);
%------------------------------------------------------------------------
%                   printing of the values on the graph
        y=0;
        for err = 1:nErr
            pc = M(meth,err)*100./T(meth);
            %lablabs = deblank(barlegend{err});
            txt = sprintf('%3.1f% \n [%s]', pc, barlegendshort{err} );

            mmm = sum(Mscaled(meth,:))/100; % 1 percent of full amount
            if M(meth,err) < mmm;
                % cannot leave too small a space for font
                if y==0
                    y = mmm;
                end
            else
                y = sum( Mscaled(meth,1:err) );
            end
    % y-position for the text
            midy = floor(y-(Mscaled(meth,err)/2));
            
    % currently the following line had to be manually edited
            xpos=placebars(meth)-0.09;
%             if pc<4
%                 % offset x-position for clarity
%                 xpos = xpos+0.1;
%             end
            text( xpos, midy, txt,'FontSize',13);

            % Draw a serious error line if necessary
            if(err==SEL)
                pc = sum( M(meth,1:SEL) )*100./T(meth);
                %txt = sprintf('%3.1f(%3.1f)', pc, sum(M(meth,1:3)) );
                %txt = sprintf('SEL:%3.1f%  ', pc );
                %text( xpos, y+100, txt, 'FontWeight','bold' );

                h=line( [xpos-0.05,xpos+0.195], [y,y] );
                set(h,'LineWidth',2);
            end
        end
% 
%         text(xpos,T(1)*(1+0.02), restxt);
%         text(xpos,T(1)*(1+0.04), infoLabel );
     end
end
    legend(barlegend(end:-1:1),-1,'Orientation','horizontal');
    set(legend,'FontWeight','bold');
    set(legend,'FontSize',12);
    set(gca,'Ylim',[0 110])
    xlabel('Different Classifiers grouped by Subjects','FontWeight','bold','FontSize',12);
    ylabel('Percentage of various measures','FontWeight','bold','FontSize',12);
    set(gca,'XTick',L);
    xLabels=['Subject 1';'Subject 2';'Subject 3';'Subject 4'];
    set(gca,'XTickLabel',xLabels,'FontWeight','bold','FontSize',12);
end
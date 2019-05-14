function separateLabels=separate(ClassLabels,scale)
    u=unique(ClassLabels);
    s=length(u);
    
    separateLabels=ClassLabels;
    
   for j=1:1:s
            separateLabels(separateLabels==u(j))=scale*(j-1);
   end
end
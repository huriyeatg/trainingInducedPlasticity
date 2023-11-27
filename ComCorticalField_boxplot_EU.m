% ComCorticalField Plot with Aboxplot

clear tra con
for i=1:4 % 5 fields
    for ii= 1:6 % for parameters; az,pitch,tim,az*tim...
        tra{ii,i} = nan(500,1);
        tra{ii,i}(1:size(datT(datT(:,7)==i,ii),1))= datT(datT(:,7)==i,ii);
        con{ii,i} = nan(500,1);
        con{ii,i}(1:size(datC(datC(:,7)==i,ii),1))=datC(datC(:,7)==i,ii);
    end
end
% Tra(parameter,field)

figure
fns = 12;

for ii=1:6
    subplot(2,3,ii)
    dt = cat(2,tra{ii,1},tra{ii,2},tra{ii,3},tra{ii,4});
    dc = cat(2,con{ii,1},con{ii,2},con{ii,3},con{ii,4});
    h = {dc;dt};
  aboxplot(h,'labels',{'A1', 'AAF','PPF','PSF'},...
      'Colormap',[1 1 1 ; 0.5 0.5 0.5]);
   ylabel('% Variance explained')
    %kruskalwallis(datC(:,ii),dIndC(:,2));
    if ii==1
       % title('Azimuth','FontSize',14)
        set(gca,'XTick',1:1:4);
        set(gca,'Ylim',[-0.1 0.30]*100);
        set(gca,'FontSize',fns,'FontWeight','bold','linewidth',1.1)
        box off
        %ylabel('Proportion of Variance Explained(%)');
    elseif ii==2
       % title('Pitch','FontSize',14)
        set(gca,'XTick',1:1:4)
        set(gca,'XTickLabel',{'A1', 'AAF','PPF','PSF'})
        set(gca,'Ylim',[-.1 0.40]*100);
        set(gca,'FontSize',fns,'linewidth',1.1,'FontWeight','bold')
        box off
    elseif ii==3
      %  title('Timbre','FontSize',14)
        set(gca,'XTick',1:1:4)
        set(gca,'XTickLabel',{'A1', 'AAF','PPF','PSF'})
        set(gca,'Ylim',[-.05 0.55]*100);
        set(gca,'FontSize',fns,'linewidth',1.1,'FontWeight','bold')
        box off
    elseif ii==4
        %title('Azimuth-Pitch','FontSize',14)
        set(gca,'XTick',1:1:4)
        set(gca,'XTickLabel',{'A1', 'AAF','PPF','PSF'})
        set(gca,'Ylim',[-.15 0.20]*100);
        set(gca,'FontSize',fns,'linewidth',1.1,'FontWeight','bold')
        box off
    elseif ii==5
       % title('Azimuth-Timbre','FontSize',14)
        set(gca,'XTick',1:1:4)
        set(gca,'XTickLabel',{'A1', 'AAF','PPF','PSF'})
        set(gca,'Ylim',[-.05 0.12]*100);
        set(gca,'FontSize',fns,'linewidth',1.1,'FontWeight','bold')
        box off
    elseif ii==6
       % title('Pitch-Timbre','FontSize',14)
        set(gca,'XTick',1:1:4)
        set(gca,'XTickLabel',{'A1', 'AAF','PPF','PSF'})
        set(gca,'Ylim',[-.07 0.25]*100);
        set(gca,'FontSize',fns,'linewidth',1.1,'FontWeight','bold')
        box off
    end
  
end

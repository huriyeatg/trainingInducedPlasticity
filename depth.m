clear all
load('/ScriptsMatlab/Analysis_2014/spikeFN_lastCorr_BFCorrected.mat')
 CoordinateCorrection
datD=[];
for k=1:length(spikeFN)
    if  spikeFN(k).driven<0.05 & ~isempty(spikeFN(k).SSRvaluesAll) & ~isnan(spikeFN(k).SSRvaluesAll)
        datD{k,1}=spikeFN(k).SSRvaluesAll(1)*100;
        datD{k,2}=spikeFN(k).SSRvaluesAll(2)*100;
        datD{k,3}=spikeFN(k).SSRvaluesAll(3)*100;
        datD{k,4}=spikeFN(k).SSRvaluesAll(4)*100;
        datD{k,5}=spikeFN(k).SSRvaluesAll(5)*100;
        datD{k,6}=spikeFN(k).SSRvaluesAll(6)*100;
        datD{k,7}=spikeFN(k).field;
        if spikeFN(k).shrank ==1 & spikeFN(k).Totalshrank==1 %32x1
            % it is 32 shrank - first half
            elect = spikeFN(k).depth*50;
        elseif spikeFN(k).shrank ==2 & spikeFN(k).Totalshrank==1 %32x1
            elect = 16*50 + spikeFN(k).depth*50;
        elseif spikeFN(k).shrank ==2   %16 x2 same for both shrank
            % some has mistakenly 5 instead of 2, typo!
            elect = spikeFN(k).depth*100;

        elseif spikeFN(k).shrank==4  % 4x1 same for all
            elect = spikeFN(k).depth*150; 
        end
        
        datD{k,8}=elect;
    end
end

dat =cell2mat(datD);

datT = dat(find(dat(:,8)<800),:);
datC = dat(find(dat(:,8)>=800),:);

% ComCorticalField_aboxplot %for all fields

% General box plot of superficial vs deeper layers for 6 
figure
set(gcf, 'PaperType', 'A4') % paper size
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'centimeters'); %necessary for the next line
set(gcf, 'PaperPosition', [1 1 24 8]);
set(gcf,'units','centimeters','position',get(gcf,'paperposition')+[0 0 0 0]); %this command shifts the 4 corners the number of units you tell it from where it is now.  the 3rd and 4th will actually alter the size 0 0 0 0 implements what you said in fig size.
set(gcf, 'Color', [1 1 1])
clear tra con

    for ii= 1:6 % for parameters; az,pitch,tim,az*tim...
        tra{ii,1} = nan(500,1);
        tra{ii,1}(1:size(datT(:,ii),1))= datT(:,ii);
        con{ii,1} = nan(500,1);
        con{ii,1}(1:size(datC(:,ii),1))=datC(:,ii);
    end

h = {cat(2,tra{:,1});cat(2,con{:,1})};
    aboxplot(h,'labels',{'Azimuth', 'Pitch','Timbre','Az-Pitch','Az-Timbre','Pit-Timbre'},...
        'Colormap',[0.5 0.5 0.5 ; 1 1 1]);
    set(gca,'Ylim',[-.1 0.30]*100);
    
    set(gca,'FontSize',12,'linewidth',1.1,'FontWeight','bold')
    box off
[h,p,stats]=ttest2(datC(:,1),datT(:,1)); p1=p; h1=h;
[h,p,stats]=ttest2(datC(:,2),datT(:,2)); p2=p; h2=h;
[h,p,stats]=ttest2(datC(:,3),datT(:,3)); p3=p; h3=h;
[h,p,stats]=ttest2(datC(:,4),datT(:,4)); p4=p; h4=h;
[h,p,stats]=ttest2(datC(:,5),datT(:,5)); p5=p; h5=h;
[h,p,stats]=ttest2(datC(:,6),datT(:,6)); p6=p; h6=h;


ttestResult=[h1,p1;h2,p2;h3,p3;h4,p4;h5,p5;h6,p6]

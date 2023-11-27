function  AllVowelConfiguration (save_path)

%% load extracted info
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

%load control metrics used in Bizley et al., 09
load control_metrics
load PropsControl

for dsa = 1:2
    if dsa==1
        DataMain = PropsControl;
        vf =struct2cell(control_metrics);
        FF = struct('field',vf(12,:));
        An = struct('animal',vf(1,:));
        DataMain = catstruct(DataMain,FF,An);
        titleSt = 'Control';
        
        datC=[]; dat=[]; t =1;
        for k=1:length(DataMain)
            if~isempty(DataMain(k).EU) & ~isnan(DataMain(k).EU)
                datC=[datC;DataMain(k).animal,DataMain(k).field]; %ideally would be animal, cortical field
                
                dat{t,1}=DataMain(k).EU(1)*100;
                dat{t,2}=DataMain(k).AI(1)*100;
                dat{t,3}=DataMain(k).EA(1)*100;
                dat{t,4}=DataMain(k).UI(1)*100;
                dat{t,5}=DataMain(k).EI(1)*100;
                dat{t,6}=DataMain(k).AU(1)*100;
                
                dat{t,7}=DataMain(k).EU(2)*100;
                dat{t,8}=DataMain(k).AI(2)*100;
                dat{t,9}=DataMain(k).EA(2)*100;
                dat{t,10}=DataMain(k).UI(2)*100;
                dat{t,11}=DataMain(k).EI(2)*100;
                dat{t,12}=DataMain(k).AU(2)*100;
                
                dat{t,13}=DataMain(k).EU(3)*100;
                dat{t,14}=DataMain(k).AI(3)*100;
                dat{t,15}=DataMain(k).EA(3)*100;
                dat{t,16}=DataMain(k).UI(3)*100;
                dat{t,17}=DataMain(k).EI(3)*100;
                dat{t,18}=DataMain(k).AU(3)*100;
                t = t+1;
            end
        end
        
    elseif dsa==2
        clear DataMain
        DataMain = trained_metrics;
        titleSt = 'Trained';
        datC=[]; dat=[]; t =1;
        for k=1:length(DataMain)
            if~isempty(DataMain(k).EU) & ~isnan(DataMain(k).EU) &...
                    (DataMain(k).animal==848 || DataMain(k).animal==832)&...
                    DataMain(k).SSRvalues_Anovatable{2,7}<0.005
                
                datC=[datC;DataMain(k).animal,DataMain(k).field]; %ideally would be animal, cortical field
                
                dat{t,1}=DataMain(k).EU(1)*100;
                dat{t,2}=DataMain(k).AI(1)*100;
                dat{t,3}=DataMain(k).EA(1)*100;
                dat{t,4}=DataMain(k).UI(1)*100;
                dat{t,5}=DataMain(k).EI(1)*100;
                dat{t,6}=DataMain(k).AU(1)*100;
                
                dat{t,7}=DataMain(k).EU(2)*100;
                dat{t,8}=DataMain(k).AI(2)*100;
                dat{t,9}=DataMain(k).EA(2)*100;
                dat{t,10}=DataMain(k).UI(2)*100;
                dat{t,11}=DataMain(k).EI(2)*100;
                dat{t,12}=DataMain(k).AU(2)*100;
                
                dat{t,13}=DataMain(k).EU(3)*100;
                dat{t,14}=DataMain(k).AI(3)*100;
                dat{t,15}=DataMain(k).EA(3)*100;
                dat{t,16}=DataMain(k).UI(3)*100;
                dat{t,17}=DataMain(k).EI(3)*100;
                dat{t,18}=DataMain(k).AU(3)*100;
                t = t+1;
            end
        end
    end
    
    dat=cell2mat(dat);
    
    
    xp1=[]; xt =[]; xx=[];
    
    xp1 = [dat(:,13),dat(:,14),dat(:,15),dat(:,16),dat(:,17),dat(:,18)];%for timbre
    xt  = [dat(:,7), dat(:,8), dat(:,9), dat(:,10),dat(:,11),dat(:,12)]; % for pitch
    xx  = [dat(:,1), dat(:,2), dat(:,3), dat(:,4), dat(:,5), dat(:,6)]; % for azimuth
    
    MeanAz =[]; MeanPt =[]; MeanTm =[];
    for ii=1:6
        MeanAz = [MeanAz;nanmean(xx(datC(:,2)==1,ii)),nanmean(xx(datC(:,2)==2,ii)),nanmean(xx(datC(:,2)==3,ii)),...
            nanmean(xx(datC(:,2)==4,ii))];
        MeanPt = [MeanPt;nanmean(xt(datC(:,2)==1,ii)),nanmean(xt(datC(:,2)==2,ii)),nanmean(xt(datC(:,2)==3,ii)),...
            mean(xt(datC(:,2)==4,ii))];
        MeanTm = [MeanTm;nanmean(xp1(datC(:,2)==1,ii)),nanmean(xp1(datC(:,2)==2,ii)),nanmean(xp1(datC(:,2)==3,ii)),...
            nanmean(xp1(datC(:,2)==4,ii))];
    end
    
    
    dx=[1;2;3;4;5;6];
    field=['  ';'EU';'AI'; 'EA';'UI';'EI';'UA'];
    
    
    figure
    fns = 12;
    
    for ii=1:3
        if ii==1
            plotdd =xx;
            MeanA = MeanAz;
            yvalue = 30;
        elseif ii==2
            plotdd =xt;
            MeanA = MeanPt;
            yvalue = 40;
        elseif ii==3
            plotdd =xp1;
            MeanA = MeanTm;
            yvalue = 60;
        end
        
        subplot(3,1,ii)
        
        aa = bar(plotdd) ; hold on
        
        set(aa(7,:),'Visible','off');
        for i = 1:size(aa,2), set(aa(:,i),'linewidth',1,'color','k'); end
        scatter(dx,MeanA(:,1),'<','filled','markerfacecolor',[0 0 0]); hold on
        scatter(dx,MeanA(:,2),'s','filled','markerfacecolor',[0.2 0.2 0.2]); hold on
        scatter(dx,MeanA(:,3),'d','filled','markerfacecolor',[0.4 0.4 0.4]); hold on
        scatter(dx,MeanA(:,4),'ok'); hold on
        
        
        set(gca,'Xlim',[0 7])
        set(gca,'XTick',0:1:6)
        set(gca,'XTickLabel', field)
        set(gca,'ylim',[-10 yvalue])
        set(gca,'YTick',-10:10:yvalue)
        set(gca,'FontSize',fns,'FontWeight','bold','linewidth',1.1)
        box off
        %legend ('A1','AAF','PPF','PSF','ADF')
    end
    
    %% Save figure
    saveas(gcf, fullfile(save_path,  ['VowelConf_',titleSt]), 'fig');
    print(gcf, '-dsvg',   fullfile(save_path, ['VowelConf_',titleSt]));
    print(gcf, '-dpng',  fullfile(save_path, ['VowelConf_',titleSt]));
end
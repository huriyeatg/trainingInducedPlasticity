function compareTrainedvsControlFeatures(save_path)

%% load metrics
load control_metrics
load PropsControl 

load trained_metrics
CoordinateCorrection

%% adjust metrics values - multiply 100 and create matrices for box plots
dat=[];
for k=1:length(trained_metrics)
    if  ~isempty(trained_metrics(k).SSRvaluesAll) & ~isnan(trained_metrics(k).SSRvaluesAll)&...
            (trained_metrics(k).animal==943 ||trained_metrics(k).animal==963 || trained_metrics(k).animal==833)
        
        dat{k,1}=trained_metrics(k).SSRvaluesAll(3)*100; % for parameters; az,pitch,tim,az*tim...
        dat{k,2}=trained_metrics(k).SSRvaluesAll(2)*100;
        dat{k,3}=trained_metrics(k).SSRvaluesAll(1)*100;
        dat{k,4}=trained_metrics(k).SSRvaluesAll(4)*100;
        dat{k,5}=trained_metrics(k).SSRvaluesAll(5)*100;
        dat{k,6}=trained_metrics(k).SSRvaluesAll(6)*100;
        dat{k,7}=trained_metrics(k).field;
    end
end

datTt=cell2mat(dat); % timbre trained animal data - Trained timbre

dat=[];
for k=1:length(trained_metrics)
    if  ~isempty(trained_metrics(k).SSRvaluesAll) & ~isnan(trained_metrics(k).SSRvaluesAll)&...
            (trained_metrics(k).animal==848 ||trained_metrics(k).animal==832)   %          (trained_metrics(k).animal==943 ||trained_metrics(k).animal==963 || trained_metrics(k).animal==833)
        
        dat{k,1}=trained_metrics(k).SSRvaluesAll(3)*100;
        dat{k,2}=trained_metrics(k).SSRvaluesAll(2)*100;
        dat{k,3}=trained_metrics(k).SSRvaluesAll(1)*100;
        dat{k,4}=trained_metrics(k).SSRvaluesAll(4)*100;
        dat{k,5}=trained_metrics(k).SSRvaluesAll(5)*100;
        dat{k,6}=trained_metrics(k).SSRvaluesAll(6)*100;
        dat{k,7}=trained_metrics(k).field;
    end
end

datTp=cell2mat(dat); % pitch trained animal data - Trained pitch


dat=[];

for k=1:length(PropsControl)
    if  ~isempty(PropsControl(k).All)  &  ~isnan(PropsControl(k).All)
        dat{k,1}=PropsControl(k).All(3)*100;
        dat{k,2}=PropsControl(k).All(2)*100;
        dat{k,3}=PropsControl(k).All(1)*100;
        dat{k,4}=PropsControl(k).All(4)*100;
        dat{k,5}=PropsControl(k).All(5)*100;
        dat{k,6}=PropsControl(k).All(6)*100;
        dat{k,7}=control_metrics(k).field;
    end
end

datC = cell2mat(dat); % control

%% plot the boxplot 
ComCorticalField_aboxplot % 3 comparsion: control vs timbre trained vs pitch trained 
    
saveas(gcf, fullfile(save_path,  'VarianceExplainedBoxPlot'), 'fig');
print(gcf, '-dsvg', fullfile(save_path, 'VarianceExplainedBoxPlot'));
print(gcf, '-dpng',   fullfile(save_path, 'VarianceExplainedBoxPlot'));
%%   ANOVA stats control vs trained timbre

all=[];
for i=1:4 % 5 fields
    for ii= 1:6 % for parameters; az,pitch,tim,az*tim...
        [h,p] = ttest2(datTt(datTt(:,7)==i,ii), datC(datC(:,7)==i,ii));
        all=[all;i,ii,h, p];
    end
end

Y = [ datC(:,1);datC(:,2); datC(:,3);...
    datTt(:,1);datTt(:,2); datTt(:,3);...
    datTp(:,1);datTp(:,2); datTp(:,3)];

lenc = size(datC(:,1),1);
lenp = size(datTp(:,1),1);
lent = size(datTt(:,1),1);
group = [ ones(lenc,1),ones(lenc,1),datC(:,7);
    ones(lenc,1),ones(lenc,1)*2,datC(:,7);
    ones(lenc,1),ones(lenc,1)*3,datC(:,7);
    ones(lent,1)*2,ones(lent,1),datTt(:,7);
    ones(lent,1)*2,ones(lent,1)*2,datTt(:,7);
    ones(lent,1)*2,ones(lent,1)*3,datTt(:,7);
    ones(lenp,1)*3,ones(lenp,1),datTp(:,7);
    ones(lenp,1)*3,ones(lenp,1)*2,datTp(:,7);
    ones(lenp,1)*3,ones(lenp,1)*3,datTp(:,7)];


[P,T,STATS,TERMS]=  anovan(Y, group,...
    'varnames',strvcat('Training', 'Timbre', 'Cortical Fields'));
[c,m,h,nms] = multcompare(STATS,'display','off');
[nms num2cell(m)];

multcompare( STATS, 'dim',1:2);

%%   ANOVA stats control vs trained pitch

clear tra con
for i=1:4 % 5 fields
    for ii= 1:6 % for parameters; az,pitch,tim,az*tim...
        tra{ii,i} = nan(500,1);
        tra{ii,i}(1:size(datTt(datTt(:,7)==i,ii),1))= datTt(datTt(:,7)==i,ii);
        tra2{ii,i} = nan(500,1);
        tra2{ii,i}(1:size(datTp(datTp(:,7)==i,ii),1))= datTp(datTp(:,7)==i,ii);
        con{ii,i} = nan(500,1);
        con{ii,i}(1:size(datC(datC(:,7)==i,ii),1))=datC(datC(:,7)==i,ii);
    end
end
ii=3% timbre
dt = cat(1,tra{ii,1},tra{ii,2},tra{ii,3},tra{ii,4});
dp = cat(1,tra2{ii,1},tra2{ii,2},tra2{ii,3},tra2{ii,4});
dc = cat(1,con{ii,1},con{ii,2},con{ii,3},con{ii,4});

Y = [ dt;dp;dc];
group = [ones(500,1), ones(500,1); ones(500,1),ones(500,1)*2;...
    ones(500,1), ones(500,1)*3; ones(500,1),ones(500,1)*4;...
    ones(500,1)*2, ones(500,1); ones(500,1)*2,ones(500,1)*2;...
    ones(500,1)*2, ones(500,1)*3; ones(500,1)*2,ones(500,1)*4;...
    ones(500,1)*3, ones(500,1); ones(500,1)*3,ones(500,1)*2;...
    ones(500,1)*3, ones(500,1)*3; ones(500,1)*3,ones(500,1)*4];

anovan(Y, group);

%% ANOVA stats for 3 groups
Y = [datC(:,2);datTt(:,2); datTp(:,2)];

group = [ones(lenc,1),datC(:,7);
    ones(lent,1)*2,datTt(:,7);
    ones(lenp,1)*3,datTp(:,7)];

[P,T,STATS,TERMS]= anovan (Y, group,...
    'varnames',strvcat('Training', 'Cortical Fields'));




function ComparisonTrainedvsUntrainedEU( save_path)

%% load extracted info
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

%load control metrics used in Bizley et al., 09
load control_metrics
load PropsControl

%%
 
datT=[];
for k=1:length(trained_metrics)
    if  ~isempty(trained_metrics(k).SSRvaluesAll) & ~isnan(trained_metrics(k).SSRvaluesAll) & (trained_metrics(k).animal==943 ||trained_metrics(k).animal==963 || trained_metrics(k).animal==833)
        datT{k,1}=trained_metrics(k).EU(1)*100;
        datT{k,2}=trained_metrics(k).EU(2)*100;
        datT{k,3}=trained_metrics(k).EU(3)*100;
        datT{k,4}=trained_metrics(k).EU(4)*100;
        datT{k,5}=trained_metrics(k).EU(5)*100;
        datT{k,6}=trained_metrics(k).EU(6)*100;
        datT{k,7}=trained_metrics(k).field;
    end
end

datT=cell2mat(datT);

datC=[];
for k=1:length(PropsControl)
    if  ~isempty(PropsControl(k).All) & ~isnan(PropsControl(k).All)
        datC{k,1}=PropsControl(k).EU(1)*100;
        datC{k,2}=PropsControl(k).EU(2)*100;
        datC{k,3}=PropsControl(k).EU(3)*100;
        datC{k,4}=PropsControl(k).EU(4)*100;
        datC{k,5}=PropsControl(k).EU(5)*100;
        datC{k,6}=PropsControl(k).EU(6)*100;
        datC{k,7}=control_metrics(k).field;
    end
end
datC=cell2mat(datC);

%% Plot the box plot
ComCorticalField_boxplot_EU 
    
%% Save figure
saveas(gcf, fullfile(save_path,  'VarianceExplainedBoxPlotForEU'), 'fig');
print(gcf, '-dsvg',   fullfile(save_path, 'VarianceExplainedBoxPlotForEU'));
print(gcf, '-dpng',   fullfile(save_path, 'VarianceExplainedBoxPlotForEU'));




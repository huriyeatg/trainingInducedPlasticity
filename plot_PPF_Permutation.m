function plot_PPF_Permutation (fig_path)

% Check if all 7 recordings are from same animal
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

dd                 = struct2cell( trained_metrics);
animalAll     = squeeze( cell2mat(dd(2,:,:)));
fieldAll          = squeeze(cell2mat(dd(11,:,:)));
penetrationAll  = squeeze(cell2mat(dd(3,:,:)));

%% Check unique animal for recordings
checkPoints  =unique(fieldAll);
fieldNames = [ {'A1'},{'AAF'},{'PPF'},{'PSF'},{'ADF'}];
values = cell(numel(checkPoints),2);
for k = 1: numel(checkPoints)
    aList = unique(animalAll(fieldAll==checkPoints(k)));
    values{k,1} = fieldNames{k};
    values{k,2} = aList;
end

% 848, 832 for  T/P GNG animals
%  833, 943, 963 for T-2AFC
% disp (values);

%% Check PPF is driven from one recording
values =[];
for k=1:length(trained_metrics)
    if  ~isempty(trained_metrics(k).SSRvaluesAll) & ~isnan(trained_metrics(k).SSRvaluesAll)&...
            (trained_metrics(k).animal==943 ||trained_metrics(k).animal==963 || trained_metrics(k).animal==833)

        values = [values; k, trained_metrics(k).field,...
            trained_metrics(k).penetration,...
            trained_metrics(k).SSRvaluesAll(3)*100];
    end
end

%df = values(values(:,2)==3,4); take PPF mean for SSR values
%nanmean(df)


% Lets do a permutation since there is only 7 recording from PPF
dd = [];
nElect = 20;
PPFArray = find (values(:,2)==3);
otherArray = find (values(:,2)~=3);
for k = 1:1000
    selectPPF = randsample(PPFArray, nElect);
    selectOther = randsample(otherArray, nElect);
    dd = [dd; nanmean(values(selectPPF,4)),nanmean(values(selectOther,4))];
end


figure
aboxplot( {dd(:,1),dd(:,2)},...
    'labels', [{'PPF'}, {'Randomly'}] , ...
    'Colormap',[0.8 0 1;0.5 0.5 0.5]);
ylabel('% Variance explained')


set(gca,'XTick',0.82:0.35:1.2)
set(gca,'XTickLabel',{'PPF selected', 'randomly selected'})
box off

pval =  signtest(dd(:,1),dd(:,2));
title(gca, pval);
saveas(gcf, fullfile(fig_path,  'TimbreCheck'), 'fig');
print(gcf, '-dsvg', fullfile(fig_path, 'TimbreCheck'));
print(gcf, '-dpng',   fullfile(fig_path, 'TimbreCheck'));
end



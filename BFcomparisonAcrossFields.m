function BFcomparisonAcrossFields
% plots Best Frequency maps for trained as well as for control dataset
warning('off', 'all')
% H Atilgan
setup_figprop;
%% load extracted info
% load trained metrics
load trained_metrics
BFcorrection
CoordinateCorrection
load gpointsTrained

%load control metrics used in Bizley et al., 09
load control_metricsStatsMore
%%  exclude nans, not significant units

% For trained metrics
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
Ftimbre = [];
animalListTimbreTrained = [943,963,833];
groupListTrained = [];
for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            ~isempty(trained_metrics(ind).SSRvaluesAll) & ~isnan(trained_metrics(ind).SSRvaluesAll)
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        %  xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        %  ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        %  indextrained = [indextrained ; ind];
        if  (ismember(trained_metrics(ind).animal, animalListTimbreTrained))
            groupListTrained = [groupListTrained, 1];
        else
            groupListTrained = [groupListTrained, 2];
        end
    end
end

% For control metrics
BFControl=[];fieldControl=[];
xControl=[]; yControl=[]; indexControl =[];
groupListControl = [];
for ind=1:length(control_metrics)
    if   ~isempty(control_metrics(ind).bf) & control_metrics(ind).field<5 & ind~=88
        BFControl      = [BFControl;control_metrics(ind).bf(1)];
        % xControl         = [xControl ; control_metrics(ind).mapx,control_metrics(ind).field];
        % yControl         = [yControl ; control_metrics(ind).mapy];
        fieldControl   = [fieldControl ; control_metrics(ind).field];
        % indexControl = [indexControl ; ind];
        groupListControl = [groupListControl,0];

    end
end

%% Plot BF histograms: trained vs control
figure
edges = 1:2:30;
titleSt = [{'A1'}; {'AAF'}; {'PPF'}; {'PSF'}];
for ind = 1:length(unique(fieldControl))
    subplot(length(unique(fieldControl)),2,ind*2-1)
    histogram(BFControl(fieldControl==ind), edges,'FaceColor','k')
    title(titleSt{ind})
    box off
    ylim([0,20])
end

for ind = 1:length(unique(fieldtrained))
    subplot(length(unique(fieldtrained)),2,ind*2)
    histogram(BFtrained(fieldtrained==ind), edges,'FaceColor','k')
    title(titleSt{ind})
    box off
    ylim([0,20])
end

% 
%% Plot BF histograms: Across training groups

% Control
figure
%colors =[0.3 0.3, 0.3), (0.8, 0, 1), (0, 0.4, 1)]
edges = 1:2:30;
titleSt = [{'A1'}; {'AAF'}; {'PPF'}; {'PSF'}];
for ind = 1:length(unique(fieldControl))
    subplot(length(unique(fieldControl)),3,ind*3-2)
    histogram(BFControl(fieldControl==ind), edges,'FaceColor','k')
    title(titleSt{ind})
    box off
    ylim([0,20])
end


% For trained timbre
animalListSelected = [943,963,833];
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
Ftimbre = [];

for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            (ismember(trained_metrics(ind).animal, animalListSelected))
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        indextrained = [indextrained ; ind];
    end
end

for ind = 1:length(unique(fieldtrained))
    subplot(length(unique(fieldtrained)),3,ind*3-1)
    histogram(BFtrained(fieldtrained==ind), edges,'FaceColor',[0.8 0 1])
    title(titleSt{ind})
    box off
    ylim([0,20])
end

% For trained timbre
animalListSelected = [943,963,833];
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
Ftimbre = [];

for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            (ismember(trained_metrics(ind).animal, animalListSelected))
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        indextrained = [indextrained ; ind];
    end
end

for ind = 1:length(unique(fieldtrained))
    subplot(length(unique(fieldtrained)),3,ind*3-1)
    histogram(BFtrained(fieldtrained==ind), edges,'FaceColor',[0.8 0 1])
    title(titleSt{ind})
    box off
    ylim([0,20])
end

% For trained timbre
animalListSelected = [848, 832];
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
Ftimbre = [];

for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            (ismember(trained_metrics(ind).animal, animalListSelected))
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        indextrained = [indextrained ; ind];
    end
end

for ind = 1:length(unique(fieldtrained))
    subplot(length(unique(fieldtrained)),3,ind*3)
    histogram(BFtrained(fieldtrained==ind), edges,'FaceColor',[0 0.4 1])
    title(titleSt{ind})
    box off
    ylim([0,20])
end

%% Run stats

% Create the data for comparison
BF = [BFtrained; BFControl];
field = [fieldtrained; fieldControl];
group = [groupListTrained, groupListControl]';

% Create a table
data = table(BF, field, group, 'VariableNames', {'BF', 'Field', 'Group'});

data.Field = categorical(data.Field);
data.Group = categorical(data.Group);
%data.Group = reordercats(data.Group, {'0', '1'});
data.Field = reordercats(data.Field, {'1', '2', '3','4'});
% Fit the mixed effects model with three groups
lme1 = fitlme(data, 'BF ~ Field * Group + (1|Field)');
disp(lme1)

%%%%%%  Organise data to compare all trained to control 
% and compare trained to control data
BF = [BFtrained; BFControl];
field = [fieldtrained; fieldControl];
group = [groupListTrained, groupListControl]';

% Create a table
data = table(BF, field, group, 'VariableNames', {'BF', 'Field', 'Group'});

data.Group(data.Group == 1 | data.Group == 2) = 1;
data.Group(data.Group == 0) = 0;% Convert Group to a categorical variable
data.Group = categorical(data.Group);
data.Field = categorical(data.Field);

% Fit the mixed effects model with combined trained group
lme3 = fitlme(data, 'BF ~ Field* Group + (Field|Group)', 'FitMethod', 'REML');
disp(lme3)

disp(unique(data.Field))
disp(unique(data.Group))

%% Compare BF to Timbre values
BFtrained=[]; fieldtrained=[];
xtrained=[]; ytrained=[];  indextrained =[];
Ftimbre = [];
for ind=1:length(trained_metrics)
    if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            ~isempty(trained_metrics(ind).SSRvaluesAll) & ~isnan(trained_metrics(ind).SSRvaluesAll)
        BFtrained       = [BFtrained;trained_metrics(ind).BF];
        xtrained          = [xtrained ; trained_metrics(ind).xcoor,trained_metrics(ind).penetration];
        ytrained          = [ytrained ; trained_metrics(ind).ycoor];
        fieldtrained   = [fieldtrained;trained_metrics(ind).field];
        indextrained = [indextrained ; ind];
        Ftimbre  = [Ftimbre; trained_metrics(ind).SSRvaluesAll(6)*100];

    end
end

figure
scatter(BFtrained,Ftimbre,'k'); hold on
ind = fieldtrained == 1;
scatter(BFtrained(ind),Ftimbre(ind),'r'); hold on
ind = fieldtrained == 2;
scatter( BFtrained(ind),Ftimbre(ind),'b'); hold on
ind = fieldtrained == 3;
scatter( BFtrained(ind),Ftimbre(ind),'g'); hold on
ind = fieldtrained == 4;
scatter( BFtrained(ind),Ftimbre(ind),'k'); hold on

xlabel('BF (Hz)')
ylabel('Timbre (% Variance explained)')

%% KS test
% Combine the trained and control data
BF = [BFtrained; BFControl];
field = [fieldtrained; fieldControl];
group = [groupListTrained, groupListControl]';

% Create a table
data = table(BF, field, group, 'VariableNames', {'BF', 'Field', 'Group'});

% Filter the data to include only the two main trained groups
data.Group(data.Group == 1 | data.Group == 2) = 1;
data.Group(data.Group == 0) = 0;% Convert Group to a categorical variable
data.Group = categorical(data.Group);
data.Field = categorical(data.Field);


% Initialize arrays to store KS test results
fields = unique(data.Field);
ks_stat = zeros(length(fields), 1);
p_value = zeros(length(fields), 1);

titleSt = [{'A1'}; {'AAF'}; {'PPF'}; {'PSF'}];
% Loop over each field and perform KS test
figure
for i = 1:length(fields)
    field = fields(i);

    % Extract BF data for the current field
    bf_trained = data.BF(data.Group == '1' & data.Field == field);
    bf_control = data.BF(data.Group == '0' & data.Field == field);

    % Perform KS test
    [h, p, ks2stat] = kstest2(bf_trained, bf_control);

    % Store KS statistic and p-value
    ks_stat(i) = ks2stat;
    p_value(i) = p;

    % Display results for the current field
    fprintf('Field %d: KS Statistic = %.4f, p-value = %.4f\n', field, ks2stat, p);

    % Plot the CDFs for the current field
    subplot(2,2,i)
    cdfplot(bf_trained);
    hold on;
    cdfplot(bf_control);
    legend('Trained', 'Control', 'Location', 'best');
    title(sprintf('CDF Comparison for Field %d', field));
    xlabel('BF');
    ylabel('CDF');
    title(titleSt{i})

    hold off;
end


% Perform multiple comparison corrections
n_comparisons = length(fields);

% Bonferroni correction
p_bonferroni = min(p_value * n_comparisons, 1);

% FDR correction using the Benjamini-Hochberg procedure
[~, ~, p_fdr,~] = fdr_bh(p_value);

% Display results with corrections
results_table = table(fields, ks_stat, p_value, p_bonferroni, p_fdr, ...
    'VariableNames', {'Field', 'KS_Statistic', 'P_Value', 'P_Bonferroni', 'P_FDR'});
disp(results_table);


%%  Prepare data for modeling
BFtrained = [];
fieldtrained = [];
xtrained = [];
ytrained = [];
indextrained = [];
Ftimbre = [];
groupListTrained = [];
animalListTimbreTrained = [943, 963, 833];

for ind = 1:length(trained_metrics)
   if  trained_metrics(ind).ToneDriven<0.05 & ~isempty(trained_metrics(ind).BF) & ~isnan(trained_metrics(ind).BF) & ...
            ~isempty(trained_metrics(ind).SSRvaluesAll) & ~isnan(trained_metrics(ind).SSRvaluesAll)

        % if trained_metrics(ind).ToneDriven < 0.05 && ...
        %         ~isempty(trained_metrics(ind).BF) && ~isnan(trained_metrics(ind).BF) && ...
        %         ~isempty(trained_metrics(ind).SSRvaluesAll) && ~isnan(trained_metrics(ind).SSRvaluesAll(1))
        BFtrained = [BFtrained; trained_metrics(ind).BF];
        xtrained = [xtrained; trained_metrics(ind).xcoor, trained_metrics(ind).penetration];
        ytrained = [ytrained; trained_metrics(ind).ycoor];
        fieldtrained = [fieldtrained; trained_metrics(ind).field];
        indextrained = [indextrained; ind];
        Ftimbre = [Ftimbre; trained_metrics(ind).SSRvaluesAll(6) * 100];

        if ismember(trained_metrics(ind).animal, animalListTimbreTrained)
            groupListTrained = [groupListTrained; 'T'];
        else
            groupListTrained = [groupListTrained; 'P'];
        end
    end
end

% Create a table
data = table(BFtrained, Ftimbre, groupListTrained, ...
    'VariableNames', {'BF', 'Ftimbre', 'Group'});

lme = fitlme(data, 'Ftimbre ~ BF * Group + (1|Group)');
disp(lme)

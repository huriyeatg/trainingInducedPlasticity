function matData = prepareTimbreData(trainingGroup, feature);
% matData = prepareSSAData(trainingGroup);
% trainingGroup : data type, can be 'Timbre', 'Pitch' , or 'Control'
% H Atilgan 28/11/2023

%% load metrics
if strcmp(trainingGroup, 'Timbre')
    animalListSelected = [943,963,833];
    load trained_metrics;
    CoordinateCorrection;
    data = trained_metrics;
elseif strcmp(trainingGroup, 'Control')
    load control_metricsStatsMore;
    dd = struct2cell(control_metrics);
    animalListSelected = unique(cell2mat(squeeze(dd(1,1,:))))';
    data = control_metrics;
    data = arrayfun(@(x) setfield(x, 'shrank', x.PE(3)), data);
    for i = 1:length(data)
        if isfield(data(i), 'bf') && ~isempty(data(i).bf) && numel(data(i).bf) >= 1
            data(i).BF = data(i).bf(1);  % Set 'BF' to the first element of 'bf'
        else
            data(i).BF = NaN;  % Set 'BF' to NaN if 'bf' doesn't exist or is empty
        end
    end

elseif strcmp(trainingGroup, 'Pitch')
    animalListSelected = [848, 832];
    load trained_metrics;
    CoordinateCorrection;
    data = trained_metrics;
elseif strcmp(trainingGroup, 'AllTrained')
    animalListSelected = [848, 832, 943,963,833];
    load trained_metrics;
    CoordinateCorrection;
    data = trained_metrics;
end

if strcmp(feature, 'Timbre')
    fea_index = 3;
elseif strcmp(feature, 'F0')
    fea_index = 2;
elseif strcmp(feature, 'Space')
    fea_index = 1;
end



%% adjust metrics values - multiply 100 and create matrices for box plots
dat = cell(length(data), 7);
% Process the data
for k = 1:length(data)
    if  ~isempty(data(k).EU)

         currentData = [data(k).EU(fea_index), data(k).AI(fea_index),data(k).EA(fea_index),...
             data(k).UI(fea_index),data(k).EI(fea_index),data(k).AU(fea_index)];
         
        if ~any(isnan(currentData)) && (ismember(data(k).animal, animalListSelected))
            dat{k,1}=(data(k).UI(fea_index)*100) +1;
            dat{k,2}=(data(k).AI(fea_index)*100) +1;
            dat{k,3}=(data(k).EU(fea_index)*100) +1;
            dat{k,4}=(data(k).EI(fea_index)*100) +1;
            dat{k,5}=(data(k).EA(fea_index)*100) +1;
            dat{k,6}=(data(k).AU(fea_index)*100) +1;
            dat{k, 7} = data(k).field;
            dat{k, 8} = data(k).shrank*data(k).animal;
            dat{k, 9} = data(k).BF;
            if any(isnan(currentData))
                disp(currentData)
            end
        end
    end
end

% Remove empty cells and convert to matrix
dat = dat(~cellfun('isempty', dat(:, 1)), :);
matData = cell2mat(dat);

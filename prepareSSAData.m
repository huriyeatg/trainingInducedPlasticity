function matData = prepareSSAData(trainingGroup);
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
    

    %% adjust metrics values - multiply 100 and create matrices for box plots
    dat = cell(length(data), 7);
    orderSSAfields = [3,2,1,4,5,6];
    % Process the data
    for k = 1:length(data)
        currentData = data(k).SSRvaluesAll;
        if ~isempty(currentData) && ~any(isnan(currentData)) && (ismember(data(k).animal, animalListSelected))
            for j = 1:6
                dat{k, j} = currentData(orderSSAfields(j)) * 100;
            end
            dat{k, 7} = data(k).field;
        end
    end

    % Remove empty cells and convert to matrix
    dat = dat(~cellfun('isempty', dat(:, 1)), :);
    matData = cell2mat(dat);

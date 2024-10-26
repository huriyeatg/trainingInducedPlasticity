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
    

    %% adjust metrics values - multiply 100 and create matrices for box plots
    dat = cell(length(data), 8);
    orderSSAfields = [3,2,1,4,5,6];
    % Process the data
    for k = 1:length(data)
        currentData = data(k).SSRvaluesAll;
        if ~isempty(currentData) && ~any(isnan(currentData)) && (ismember(data(k).animal, animalListSelected)) %&& ~isnan(data(k).BF)
            for j = 1:6
                dat{k, j} = (currentData(orderSSAfields(j)) * 100)+1 ; % 1 is added to avoid 0 values for poisson links & really low values for log links
            end
            dat{k, 7} = data(k).field;
            dat{k, 8} = data(k).shrank*data(k).animal; % unique penetration for each animal
            dat{k, 9} = data(k).BF;
        end
    end

    % Remove empty cells and convert to matrix
    dat = dat(~cellfun('isempty', dat(:, 1)), :);
    matData = cell2mat(dat);

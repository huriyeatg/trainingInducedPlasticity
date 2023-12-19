function matData = extractData(trainingGroup,fieldName, cellbycell)
% matData = extractData(Traintype,fieldName)
% trainGroup : data type, can be 'Timbre', 'Pitch' , AllTrained' or 'Control'
% H Atilgan 03/12/2023

if nargin < 3
   cellbycell = false;
end

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
dat = cell(length(data), 1);
% Process the data
for k = 1:length(data)
    if (ismember(data(k).animal, animalListSelected))

        % if strcmp(type, 'Timbre') || strcmp(type, 'Pitch')
        %     currentData = data(k).SSRvaluesAll;
        % else
        %     currentData = PropsControl(k).All;
        % end
        if strcmp(trainingGroup, 'Control')
            dat{k,1} = full(data(k).(fieldName));
        else
            dat{k,1} = data(k).(fieldName);
        end
    end

end

if cellbycell == false
    matData = cell2mat(dat);
else
    matData = dat;
end

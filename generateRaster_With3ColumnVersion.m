function generateRaster_With3ColumnVersion
%Load your data
setup_figprop
load('D:\trainingInducedPlasticity\info_data\spikeData_trained.mat')

% Create a folder to save the raster plots if it doesn't exist
outputFolder = 'D:\trainingInducedPlasticity\rasterPlots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

load trained_metrics;
% Number of cells
%selectedCells = [62    85    89   165   183   270   284   306   329   471   495   508   565   584, 661   664   667   681   689   690   738   742   788];
%numCells = numel(selectedCells);
% Loop through each cell
for ind = 1:791
    disp(ind)
    % Extract spikes and stim information for the current cell
    cellID = ind; %selectedCells(ind);
    spikes = spikeData(cellID).spikes;
    stim = spikeData(cellID).stim;
    animal = trained_metrics(spikeData(cellID).index).animal;
    field =  trained_metrics(spikeData(cellID).index).field;
    titleStInfo = sprintf('Animal: %d, Field: %d', animal, field);

    % Get unique stimulus combinations and their indices
    [uniqueStim, ~, stimIndices] = unique(stim, 'rows');

    % Number of unique stimuli
    numUniqueStim = size(uniqueStim, 1);

    % Create a new figure for the raster plot
    figure('Visible', 'off');
    hold on;

    % Create Y-axis tick labels for stimulus IDs and store plot data
    yTickLabels = cell(numUniqueStim, 1);
    stimInfo = cell(numUniqueStim, 1); % To store the stimulus information for sorting
    plotData = cell(numUniqueStim, 1); % To store the spike plot data

    for stimID = 1:numUniqueStim
        trialIndices = find(stimIndices == stimID);
        if ~isempty(trialIndices)
            % Generate label with detailed stimulus information
            F0 = uniqueStim(stimID, 2);
            F1 = uniqueStim(stimID, 3);
            F2 = uniqueStim(stimID, 4);
            Az = uniqueStim(stimID, 1);
            stimInfo{stimID} = [F0, F1, F2, Az];
            yTickLabels{stimID} = sprintf('%d, %d, %d', F0, F1, Az);

            % Collect spike plot data
            spikeTimesPerTrial = {};
            for trialIdx = 1:length(trialIndices)
                trialID = trialIndices(trialIdx);
                spikeTimes = find(spikes(trialID, :));
                spikeTimesPerTrial{trialIdx} = spikeTimes;
            end
            plotData{stimID} = spikeTimesPerTrial;
        end
    end

    % Convert cell array to matrix for sorting
    stimInfoMatrix = cell2mat(stimInfo);

    % Lets sort them 3 times, and plot 3 columns
    sortNum = size(stimInfoMatrix,2);
    index = [ 1 2 4];
    titleSt = ['F0'; 'F1'; 'Az'];
    yTickLabelsAll = yTickLabels;
    plotDataAll = plotData;
    for kk = 1:(sortNum-1)
        plotInd = index(kk);
        subplot(1,sortNum-1,kk); hold on

        % Sort yTickLabels and plotData based on F1 values (3rd column)
        [~, sortIdx] = sortrows(stimInfoMatrix, [plotInd]);
        yTickLabels = yTickLabelsAll(sortIdx);
        plotData = plotDataAll(sortIdx);

        % Plot the sorted raster data
        currentTrialIdx = 1;
        yTicks = zeros(1, numUniqueStim); % Initialize yTicks
        for sortedStimID = 1:numUniqueStim
            spikeTimesPerTrial = plotData{sortedStimID};
            numTrials = length(spikeTimesPerTrial);
            % Position the label in the middle of the trials
            yTicks(sortedStimID) = currentTrialIdx + (numTrials - 1) / 2;
            for trialIdx = 1:numTrials
                spikeTimes = spikeTimesPerTrial{trialIdx};
                plot(spikeTimes, repmat(currentTrialIdx, size(spikeTimes)), 'k.', 'MarkerSize', 5);
                currentTrialIdx = currentTrialIdx + 1;

            end
        end
        ymax = size(spikes,1);
        xmax = 500;
        title (titleSt(kk,:))
        xlabel('Time (ms)');
        set(gca, 'YTick', yTicks, 'YTickLabel', yTickLabels, 'TickLabelInterpreter', 'none',  'FontSize', 8);
        ylim([0,ymax])
        xlim([0,xmax])
        if kk ==1
              ylabel('Trial (sorted repeatedly)');
        end

    end
    sgtitle(titleStInfo)
    
    data(ind).stim = stim;
    data(ind).spikes = spikes;


    % Format plot
  %  title(['Raster Plot for Cell ' num2str(cellID)]);
    
    hold off;

    % Save the figure
    saveas(gcf, fullfile(outputFolder, ['Cell_' num2str(cellID) '_raster.png']));
    set(gcf, 'Renderer', 'painters'); % Try 'painters', 'opengl', or 'zbuffer'
    saveas(gcf, 'figure.svg');

    % Close the figure
    close(gcf);
end

save dataForRaster data


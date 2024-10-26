function generateRaster
%Load your data
setup_figprop
load('D:\trainingInducedPlasticity\info_data\spikeData_trained.mat')

% Create a folder to save the raster plots if it doesn't exist
outputFolder = 'D:\trainingInducedPlasticity\raster_plots';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Number of cells
numCells = numel(spikeData);

% Loop through each cell
for cellID = 51:numCells
    % Extract spikes and stim information for the current cell
    spikes = spikeData(cellID).spikes;
    stim = spikeData(cellID).stim;
    
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

    % Sort yTickLabels and plotData based on F2 values (3rd column)
    [~, sortIdx] = sortrows(stimInfoMatrix, [2, -1, -3]);
    yTickLabels = yTickLabels(sortIdx);
    plotData = plotData(sortIdx);
    
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
    
    % Set the Y-axis labels
    set(gca, 'YTick', yTicks, 'YTickLabel', yTickLabels, 'TickLabelInterpreter', 'none',  'FontSize', 8);
    
    % Format plot
    title(['Raster Plot for Cell ' num2str(cellID)]);
    xlabel('Time (ms)');
    ylabel('Trial');
    
    hold off;
    
    % Save the figure
    saveas(gcf, fullfile(outputFolder, ['Cell_' num2str(cellID) '_raster.png']));
    set(gcf, 'Renderer', 'painters'); % Try 'painters', 'opengl', or 'zbuffer'
saveas(gcf, 'figure.svg');
    
    % Close the figure
    close(gcf);
end

% Notes
% Example cell ID - 471, F0943 Timbre  (297 more spiky version)

% 109 more specific to one combination, 111, 223
% 115,121, 245 F2 ?
% 297, 369 400 402
% 538 550
169
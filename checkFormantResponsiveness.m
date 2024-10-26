function checkFormantResponsiveness

%%%%% PCA analysis Load your data (assuming 'spikeData' structure with 'resp', 'spon', and 'stim')
load('D:\trainingInducedPlasticity\info_data\control_metricsStatsMore.mat')
spikeData = control_metrics;

% Initialize variablestrained_metrics
numCells = numel(spikeData);
respAll = [];
stimAll = [];

% Collect all responses and corresponding stimuli
for cellID = 1:numCells
    resp = full(spikeData(cellID).resp);
    stim = spikeData(cellID).stim;
    respAll = [respAll; resp];
    stimAll = [stimAll; stim];
end

% Standardize the response data
respAll = zscore(respAll);

% Perform PCA
[coeff, score, latent, tsquared, explained, mu] = pca(respAll);

% Plot the explained variance by each principal component
figure;
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Variance Explained by Principal Components');

% Plot the first two principal components
figure;
gscatter(score(:, 1), score(:, 2), stimAll(:,3)+stimAll(:, 4)); % Assuming column 2 is the formant frequency
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA of Neural Responses Colored by Formant Frequency');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Calculate Mutual Information for control 
figure 
subplot(3,1,1)
load('D:\trainingInducedPlasticity\info_data\control_metricsStatsMore.mat')
spikeData = control_metrics;

% Initialize variablestrained_metrics
numCells = numel(spikeData);
respAll = [];
stimAll = [];

% Collect all responses and corresponding stimuli
for cellID = 1:numCells
    resp = full(spikeData(cellID).resp);
    stim = spikeData(cellID).stim;
    respAll = [respAll; resp];
    stimAll = [stimAll; stim];
end

numBins = 10;
miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);
miDistances = mutualinfo(respAll, abs(diff(stimAll(:, 2:3), 1, 2)), numBins);
miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);
bandwidths = max(stimAll(:, 2:3), [], 2) - min(stimAll(:, 2:3), [], 2);
miSpectralIntegration_bandwidth = mutualinfo(respAll, bandwidths, numBins);

miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
miSpectralIntegration_sum2 = mutualinfo(respAll, mean(stimAll(:, 3:4), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);


base = 0;%mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);

% Plot Mutual Information results
bar([miAz, miF0, miF1, miF2,...
    miSpectralIntegration_sum, miSpectralIntegration_harmonic, miSpectralIntegration_geometric,...
    miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*1000-base, 'FaceColor','k');
set(gca, 'XTickLabel', []);%{'Az', 'F0', 'F1','F2', 'Relative Distances', 'SI sum', 'SI harmonic', 'SI geometric', 'SI bandwith', 'SI sum', 'SI harmonic', 'SI geometric', 'SI bandwith'});
ylabel('Mutual Information');
title('Control');
box off

%% %% 
subplot(3,1,2)
tp_disc_animalIDs = [848, 832]; % Example animal IDs for TP-Disc group
t_id_animalIDs = [943, 963, 833]; % Example animal IDs for T-ID group
load('D:\trainingInducedPlasticity\info_data\trained_metrics.mat')
spikeDataAll = trained_metrics;

% % Separate the data into TP-Disc and T-ID groups
spikeData = spikeDataAll(ismember([spikeDataAll.animal], tp_disc_animalIDs));

% Initialize variablestrained_metrics
numCells = numel(spikeData);
respAll = [];
stimAll = [];

% Collect all responses and corresponding stimuli
for cellID = 1:numCells
    resp = full(spikeData(cellID).resp);
    stim = spikeData(cellID).stim;
    respAll = [respAll; resp];
    stimAll = [stimAll; stim];
end

numBins = 10;
miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);


miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);

miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
miSpectralIntegration_sum2 = mutualinfo(respAll, sum(stimAll(:, 3:4), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);


base = 0;% mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);

% Plot Mutual Information results
bar([miAz, miF0, miF1, miF2, ...
    miSpectralIntegration_sum,miSpectralIntegration_harmonic,miSpectralIntegration_geometric,...
    miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*1000-base, 'FaceColor','k');

ylabel('Mutual Information');
title('Trained - TD-Disc');
box off


%% %%%%
subplot(3,1,3)
% % Separate the data into TP-Disc and T-ID groups
spikeData = spikeDataAll(ismember([spikeDataAll.animal], t_id_animalIDs));

% Initialize variablestrained_metrics
numCells = numel(spikeData);
respAll = [];
stimAll = [];

% Collect all responses and corresponding stimuli
for cellID = 1:numCells
    resp = full(spikeData(cellID).resp);
    stim = spikeData(cellID).stim;
    respAll = [respAll; resp];
    stimAll = [stimAll; stim];
end

numBins = 10;
miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);


miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);

miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
miSpectralIntegration_sum2 = mutualinfo(respAll, sum(stimAll(:, 3:4), 2), numBins);
HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);


base = 0;% mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);

% Plot Mutual Information results
bar([miAz, miF0, miF1, miF2, ...
    miSpectralIntegration_sum,miSpectralIntegration_harmonic,miSpectralIntegration_geometric,...
    miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*1000-base, 'FaceColor','k');
set(gca, 'XTickLabel', {'Az', 'F0', 'F1','F2', 'SI sum', 'SI harmonic', 'SI geometric', 'SI Distances','SI sum', 'SI harmonic', 'SI geometric', 'SI bandwith'});
xlabel('Stimulus Feature');
ylabel('Mutual Information');
title('Trained T-ID');
box off

% Save figures
saveas(gcf, 'mutual_information_analysis.png');

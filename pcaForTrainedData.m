function pcaForTrainedData

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


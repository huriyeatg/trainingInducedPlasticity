function plotMutualInformationBarPlotsWithSubSample

numIterations = 1000;
numSamples = 1e5;% Equalize sample size


% Plot control data
figure;
subplot(3,1,1)
data = load('D:\trainingInducedPlasticity\info_data\control_metricsStatsMore.mat');
spikeData = data.control_metrics;

plotMutualInformationBarPlots
hBar.FaceColor = 'k';  % Ensure it's set to flat mode
title('Control');
box off

%%
tp_disc_animalIDs = [848, 832]; % Example animal IDs for TP-Disc group
t_id_animalIDs = [943, 963, 833]; % Example animal IDs for T-ID group
load('D:\trainingInducedPlasticity\info_data\trained_metrics.mat')
spikeDataAll = trained_metrics;

% Plot for T-ID groups
subplot(3,1,2)
spikeData = spikeDataAll(ismember([spikeDataAll.animal], t_id_animalIDs));
plotMutualInformationBarPlots
hBar.FaceColor = 'm';  % Ensure it's set to flat mode
title('Trained T-ID');

% Plot for TP-Disc
subplot(3,1,3)
spikeData = spikeDataAll(ismember([spikeDataAll.animal], tp_disc_animalIDs));
plotMutualInformationBarPlots
hBar.FaceColor = 'b';  % Ensure it's set to flat mode
title('Trained - TD-Disc');


%%
saveas(gcf, 'mutual_information_analysisWithSubPopulation.svg');
set(gcf, 'Renderer', 'painters'); % Try 'painters', 'opengl', or 'zbuffer'
saveas(gcf, 'mutual_information_analysisWithSubPopulation.svg');


%%

% % Function to calculate mutual information
% function mi = mutualinfo(x, y, numBins)
%     % Discretize the data
%     xDiscretized = discretize(x, numBins);
%     yDiscretized = discretize(y, numBins);
%
%     % Compute the joint histogram
%     jointHist = accumarray([xDiscretized, yDiscretized], 1, [], @sum, 0);
%
%     % Convert joint histogram to joint probability distribution
%     jointProb = jointHist / sum(jointHist(:));
%
%     % Compute marginal probabilities
%     xProb = sum(jointProb, 2);
%     yProb = sum(jointProb, 1);
%
%     % Compute entropies
%     entropyX = -sum(xProb .* log2(xProb + eps));
%     entropyY = -sum(yProb .* log2(yProb + eps));
%     jointEntropy = -sum(jointProb(:) .* log2(jointProb(:) + eps));
%
%     % Compute mutual information
%     mi = entropyX + entropyY - jointEntropy;
% end



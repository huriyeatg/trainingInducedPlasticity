% Initialize variables
numCells = numel(spikeData);
respAll = [];
stimAll = [];

% Collect all responses and corresponding stimuli
for cellID = 1:numCells
    resp = full(spikeData(cellID).resp);
    stim = spikeData(cellID).stim;
    stim = calculate_harmonics(stim) ; % lets calculate harmonics as 5th and 6th column
    
    respAll = [respAll; resp];
    stimAll = [stimAll; stim];
 
end


% MI for main formants/features
[miAz, varAz] = subsampled_mutualinfo(respAll, stimAll(:, 1), numBins, numSamples, numIterations);
[miF0, varF0] = subsampled_mutualinfo(respAll, stimAll(:, 2), numBins, numSamples, numIterations);
[miF1, varF1] = subsampled_mutualinfo(respAll, stimAll(:, 3), numBins, numSamples, numIterations);
[miF2, varF2] = subsampled_mutualinfo(respAll, stimAll(:, 4), numBins, numSamples, numIterations);

% MI for F0 - F1 integration
Sum_F0_F1 = sum(stimAll(:, [2,3]), 2);
[miSpectralIntegration_sum, var_sum] = subsampled_mutualinfo(respAll, Sum_F0_F1, numBins, numSamples, numIterations);

HarmonicMean_F0_F1 = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
[miSpectralIntegration_harmonic, var_har] = subsampled_mutualinfo(respAll, HarmonicMean_F0_F1, numBins, numSamples, numIterations);

GeometricMean_F0_F1 = sqrt(stimAll(:, 2) .* stimAll(:, 3));
[miSpectralIntegration_geometric, var_geo] = subsampled_mutualinfo(respAll, GeometricMean_F0_F1, numBins, numSamples, numIterations);

% MI for F1 - F2 integration
Sum_F1_F2 = sum(stimAll(:, [3,4]), 2);
[miSpectralIntegration_sum2, var_sum2] = subsampled_mutualinfo(respAll, Sum_F1_F2, numBins, numSamples, numIterations);

HarmonicMean_F1_F2 = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
[miSpectralIntegration_harmonic2, var_har2] = subsampled_mutualinfo(respAll, HarmonicMean_F1_F2, numBins, numSamples, numIterations);

GeometricMean_F1_F2 = sqrt(stimAll(:, 3) .* stimAll(:, 4));
[miSpectralIntegration_geometric2, var_geo2] = subsampled_mutualinfo(respAll, GeometricMean_F1_F2, numBins, numSamples, numIterations);

[miDistances2,varDistances2] = subsampled_mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins, numSamples, numIterations);

% Bandwidths calculation (if necessary)
% bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
% miSpectralIntegration_bandwidth2 = subsampled_mutualinfo(respAll, bandwidths, numBins, numSamples, numIterations);

% Base MI calculation for comparison
base = 0; % Change this part if necessary for base mutual information calculation


% Collect MI values and their confidence intervals
MI_values = [miAz, miF0, miF1, miF2, ...
    miSpectralIntegration_sum, miSpectralIntegration_harmonic, miSpectralIntegration_geometric, ...
    miSpectralIntegration_sum2, miSpectralIntegration_harmonic2, miSpectralIntegration_geometric2, miDistances2];

% Collect confidence intervals from 'subsampled_mutualinfo' outputs
% Each 'var' variable contains the lower and upper bounds of the confidence interval (muci)
CI_lower = [varAz(1), varF0(1), varF1(1), varF2(1), ...
    var_sum(1), var_har(1), var_geo(1), ...
    var_sum2(1), var_har2(1), var_geo2(1), varDistances2(1)];

CI_upper = [varAz(2), varF0(2), varF1(2), varF2(2), ...
    var_sum(2), var_har(2), var_geo(2), ...
    var_sum2(2), var_har2(2), var_geo2(2), varDistances2(2)];

% Calculate error bar lengths (differences between mean and confidence interval bounds)
Error_lower = MI_values - CI_lower;
Error_upper = CI_upper - MI_values;

% Create the bar graph with error bars
hBar = bar(MI_values, 'FaceColor', 'k');  % Plot the MI values

hold on;

% Add error bars using the errorbar function
numBars = length(MI_values);
x = 1:numBars;  % X positions of the bars

% Error bars can be asymmetric, so we use the 'errorbar' function with both lower and upper errors
errorbar(x, MI_values, Error_lower, Error_upper, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);

hold off;

% Customize the plot
set(gca, 'XTick', x);
set(gca, 'XTickLabel', {'Az', 'F0', 'F1', 'F2', 'SI sum', 'SI harmonic', 'SI Geometric', 'SI sum', 'SI harmonic', 'SI Geometric', 'Difference'});
xtickangle(45);  % Rotate x-axis labels if they overlap
ylabel('MI');

% Optionally, adjust y-axis limits for better visibility
%ylim([0, max(MI_values + Error_upper) * 1.1]);
box off



%% OLD CODE
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
% % end
% 
% %%%% NO SUBSAMPLED VERSION OF THE CODE
% % Plot control data
% figure;
% subplot(3,1,1)
% data = load('D:\trainingInducedPlasticity\info_data\control_metricsStatsMore.mat');
% spikeData = data.control_metrics;
% 
% % Initialize variables
% numCells = numel(spikeData);
% respAll = [];
% stimAll = [];
% 
% % Collect all responses and corresponding stimuli
% for cellID = 1:numCells
%     resp = full(spikeData(cellID).resp);
%     stim = spikeData(cellID).stim;
%     respAll = [respAll; resp];
%     stimAll = [stimAll; stim];
% end
% 
% numBins = 10;
% miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
% miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
% miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
% miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);
% miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
% miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
% miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);
% miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
% miSpectralIntegration_sum2 = mutualinfo(respAll, sum(stimAll(:, 3:4), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
% miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
% miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
% bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
% miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);
% base = 0;% mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);
% 
% % Plot Mutual Information results
% bar([miAz, miF0, miF1, miF2, ...
%     miSpectralIntegration_sum,miSpectralIntegration_harmonic,miSpectralIntegration_geometric,...
%     miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*6000, 'FaceColor','k');
% ylabel('MI');
% ylim([0 0.4])
% title('Control');
% box off
% 
% 
% %%
% subplot(3,1,2)
% tp_disc_animalIDs = [848, 832]; % Example animal IDs for TP-Disc group
% t_id_animalIDs = [943, 963, 833]; % Example animal IDs for T-ID group
% load('D:\trainingInducedPlasticity\info_data\trained_metrics.mat')
% spikeDataAll = trained_metrics;
% % % Separate the data into TP-Disc and T-ID groups
% spikeData = spikeDataAll(ismember([spikeDataAll.animal], tp_disc_animalIDs));
% % Initialize variablestrained_metrics
% numCells = numel(spikeData);
% respAll = [];
% stimAll = [];
% % Collect all responses and corresponding stimuli
% for cellID = 1:numCells
%     resp = full(spikeData(cellID).resp);
%     stim = spikeData(cellID).stim;
%     respAll = [respAll; resp];
%     stimAll = [stimAll; stim];
% end
% numBins = 10;
% miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
% miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
% miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
% miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);
% miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
% miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
% miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);
% miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
% miSpectralIntegration_sum2 = mutualinfo(respAll, sum(stimAll(:, 3:4), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
% miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
% miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
% bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
% miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);
% base = 0;% mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);
% % Plot Mutual Information results
% bar([miAz, miF0, miF1, miF2, ...
%     miSpectralIntegration_sum,miSpectralIntegration_harmonic,miSpectralIntegration_geometric,...
%     miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*1000-base, 'FaceColor','k');
% ylabel('MI');
% ylim([0 0.4])
% title('Trained - TD-Disc');
% box off
% 
% %%
% subplot(3,1,3)
% % % Separate the data into TP-Disc and T-ID groups
% spikeData = spikeDataAll(ismember([spikeDataAll.animal], t_id_animalIDs));
% % Initialize variablestrained_metrics
% numCells = numel(spikeData);
% respAll = [];
% stimAll = [];
% % Collect all responses and corresponding stimuli
% for cellID = 1:numCells
%     resp = full(spikeData(cellID).resp);
%     stim = spikeData(cellID).stim;
%     respAll = [respAll; resp];
%     stimAll = [stimAll; stim];
% end
% numBins = 10;
% miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
% miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
% miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
% miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);
% miSpectralIntegration_sum = mutualinfo(respAll, sum(stimAll(:, 2:3), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 2) + 1./stimAll(:, 3));
% miSpectralIntegration_harmonic = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 2) .* stimAll(:, 3));
% miSpectralIntegration_geometric = mutualinfo(respAll, geometricMean, numBins);
% miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
% miSpectralIntegration_sum2 = mutualinfo(respAll, sum(stimAll(:, 3:4), 2), numBins);
% HarmonicMean = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
% miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean, numBins);
% geometricMean = sqrt(stimAll(:, 3) .* stimAll(:, 4));
% miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean, numBins);
% bandwidths = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
% miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths, numBins);
% base = 0;% mutualinfo([respAll;respAll;respAll;respAll], [stimAll(:, 1);stimAll(:, 2);stimAll(:, 3);stimAll(:, 4)], numBins);
% % Plot Mutual Information results
% bar([miAz, miF0, miF1, miF2, ...
%     miSpectralIntegration_sum,miSpectralIntegration_harmonic,miSpectralIntegration_geometric,...
%     miDistances2, miSpectralIntegration_sum2,miSpectralIntegration_harmonic2,miSpectralIntegration_geometric2,miSpectralIntegration_bandwidth2]*1000-base, 'FaceColor','k');
% set(gca, 'XTickLabel', {'Az', 'F0', 'F1','F2', 'SI sum', 'SI harmonic', 'SI geometric', 'SI Distances','SI sum', 'SI harmonic', 'SI geometric', 'SI bandwith'});
% xlabel('Stimulus Feature');
% ylabel('MI');
% ylim([0 0.4])
% title('Trained T-ID');
% box off
% 
% saveas(gcf, 'mutual_information_analysis.svg');
% set(gcf, 'Renderer', 'painters'); % Try 'painters', 'opengl', or 'zbuffer'
% saveas(gcf, 'figure.svg');
% 
% 
% %%
% 
% 
% % Function to process a group of cells and compute mutual information
%     function mi_values = process_group(spikeData, numBins)
%         numCells = numel(spikeData);
%         respAll = [];
%         stimAll = [];
% 
%         for cellID = 1:numCells
%             resp = full(spikeData(cellID).resp);
%             stim = spikeData(cellID).stim;
%             respAll = [respAll; resp];
%             stimAll = [stimAll; stim];
%         end
% 
%         miAz = mutualinfo(respAll, stimAll(:, 1), numBins);
%         miF0 = mutualinfo(respAll, stimAll(:, 2), numBins);
%         miF1 = mutualinfo(respAll, stimAll(:, 3), numBins);
%         miF2 = mutualinfo(respAll, stimAll(:, 4), numBins);
%         miDistances2 = mutualinfo(respAll, abs(diff(stimAll(:, 3:4), 1, 2)), numBins);
%         miSpectralIntegration_sum2 = mutualinfo(respAll, mean(stimAll(:, 3:4), 2), numBins);
%         HarmonicMean2 = 2 ./ (1./stimAll(:, 3) + 1./stimAll(:, 4));
%         miSpectralIntegration_harmonic2 = mutualinfo(respAll, HarmonicMean2, numBins);
%         geometricMean2 = sqrt(stimAll(:, 3) .* stimAll(:, 4));
%         miSpectralIntegration_geometric2 = mutualinfo(respAll, geometricMean2, numBins);
%         bandwidths2 = max(stimAll(:, 3:4), [], 2) - min(stimAll(:, 3:4), [], 2);
%         miSpectralIntegration_bandwidth2 = mutualinfo(respAll, bandwidths2, numBins);
% 
%         mi_values = [miAz, miF0, miF1, miF2, ...
%             miDistances2, miSpectralIntegration_sum2, miSpectralIntegration_harmonic2, miSpectralIntegration_geometric2, miSpectralIntegration_bandwidth2] ;
%     end
% end
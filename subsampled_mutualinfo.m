function [miSubsampleMean, var] = subsampled_mutualinfo(respAll, stimAll, numBins, numSamples, numIterations)
    % Initialize array to store MI for each iteration
    miSubsample = nan(numIterations, 1);
    
    n = size(respAll, 1); % Total number of samples
    
    for i = 1:numIterations
        % Randomly sample with replacement
        sampleIdx = randsample(n, numSamples, false);
        
        % Get subsampled responses and stimuli
        respSample = respAll(sampleIdx, :);
        stimSample = stimAll(sampleIdx, :);
        
        % Calculate mutual information on the subsample
        miSubsample(i) = mutualinfo(respSample, stimSample, numBins);
    end
    
    % Return the average MI and its variance
    [mu, sigma, muci, sigmaci] = normfit(miSubsample);
    miSubsampleMean = mu;
    var = muci;
    
    fprintf('Mean MI: %.4f, CI_upper: %.4f\n', miSubsampleMean, muci(1));
end

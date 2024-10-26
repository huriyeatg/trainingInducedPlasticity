function mi = mutualinfo(x, y, numBins)
    x = discretize(x, numBins);
    y = discretize(y, numBins);
    jointHist = accumarray([x y], 1);
    jointProb = jointHist / sum(jointHist(:));
    xProb = sum(jointProb, 2);
    yProb = sum(jointProb, 1);
    entropyX = -sum(xProb .* log2(xProb + eps));
    entropyY = -sum(yProb .* log2(yProb + eps));
    jointEntropy = -sum(jointProb(:) .* log2(jointProb(:) + eps));
    mi = (entropyX + entropyY - jointEntropy) *100;
end
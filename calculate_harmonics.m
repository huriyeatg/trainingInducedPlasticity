function stim_with_harmonics = calculate_harmonics(stim)
    % Input: stim matrix where:
    % 1st column: Az
    % 2nd column: F0
    % 3rd column: F1
    % 4th column: F2
    %
    % Output: stim_with_harmonics matrix where:
    % 5th column: H1 (closest harmonic to F1)
    % 6th column: H2 (closest harmonic to F2)

    % Initialize the output matrix with additional columns for harmonics
    stim_with_harmonics = stim;

    % Iterate over each row in the stim matrix
    for i = 1:size(stim, 1)
        % Extract F0, F1, and F2 for the current row
        F0 = stim(i, 2);  
        F1 = stim(i, 3);  
        F2 = stim(i, 4);  

        % Set harmonics for F0 and find closest harmonic for F1 and F2
        numHarmonics = 10;  % Arbitrary number of harmonics to calculate
        harmonicsF0 = F0 * (1:numHarmonics);  % Harmonics of F0

        % Find the closest harmonic to F1
        [~, idx1] = min(abs(harmonicsF0 - F1));
        H1 = harmonicsF0(idx1);  

        % Find the closest harmonic to F2
        [~, idx2] = min(abs(harmonicsF0 - F2));
        H2 = harmonicsF0(idx2);  

        % Append the harmonics to the stim matrix
        stim_with_harmonics(i, 5) = H1;  % 5th column: closest harmonic to F1
        stim_with_harmonics(i, 6) = H2;  % 6th column: closest harmonic to F2
    end
end

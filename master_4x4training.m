% Master Analysis - for trained vs controlled data of 64 vowel stimuli
% H Atilgan 7 Jan 2022

clear all
clc
setup_figprop
root_path = '/Volumes/GoogleDrive/My Drive/Code/4x4training';
fig_path   = fullfile(root_path, 'figs');

%% Preprocessing:  To generate a structure that includes all metrics for all the figures below
generateMetricsForAllUnits

%% Analyses for the project

% check the info in training dataset
info_check

% Normalised spike rate is missing - Figure Space C
% Vowel pair variance explained figure is missing - Figure timbre E

% To generate the BF map
plotBestFrequencyMaps(fig_path) % in figure 1

% To generate the box plot for each field
compareTrainedvsControlFeatures(fig_path)% stats done!

% To generate the voronoin maps of cortex distribution
corticalMap (fig_path)

% To generate all the different vowel configuration
AllVowelConfiguration(fig_path)

%% Extra analysis - not reported

% To generate the trained vowel vs untrained vowel
ComparisonTrainedvsUntrainedEU (fig_path)

%To look at the depth differences
depth % stats done!

% For freq figure of vowels
plot_vowelFormants(fig_path)
   
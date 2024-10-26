% Master Analysis - for trained vs controlled data of 64 vowel stimuli
% H Atilgan 7 Jan 2022

clear all
clc
setup_figprop
root_path = 'C:\Users\Huriye\Documents\code\trainingInducedPlasticity';
data_path = 'C:\Users\Huriye\Documents\code\trainingInducedPlasticity\info_data'
fig_path   = fullfile(root_path, 'figs');


%% Preprocessing:  To generate a structure that includes all metrics for all the figures below
generateMetricsForAllUnits

%% Analyses for the project

% check the info in training dataset
plot_PPF_Permutation (fig_path)

% Normalised spike rate is missing - Figure Space C
% Vowel pair variance explained figure is missing - Figure timbre E

% To generate the BF map
plotBestFrequencyMaps(fig_path) % in figure 1

% To generate the box plot for each field
compareTrainedvsControlFeatures(fig_path) % stats done!

% To generate the voronoin maps of cortex distribution
corticalMap (fig_path)

% To generate all the different vowel configuration
AllVowelConfiguration(fig_path)

%% Reviewer asked analysis: Raw data analysi
generateRaster
generateRaster_With3ColumnVersion

pcaForTrainedData
%checkFormantResponsiveness % PCA analysis
BFcomparisonAcrossFields % For review, check BF maps

%plotMutualInformationBarPlots
plotMutualInformationBarPlotsWithSubSample
%% Extra analysis - not reported

% To generate the trained vowel vs untrained vowel
ComparisonTrainedvsUntrainedEU (fig_path)

%To look at the depth differences
depth % stats done!

% For freq figure of vowels
plot_vowelFormants(fig_path)
   

%% Required for harmonics 
fs=10000;
artVowel = newMakeVowel(0.3, fs, 200, 730,2058,2979,4294); %newMakeVowel(dur,sampleRate, F0, f1,f2,f3,f4)
plot(artVowel)
% /a/: F1–F4 at 936, 1551, 2815, and 4290 Hz; 
% /ε/: 730, 2058, 2979, and 4294 Hz; 
% /u/: 460, 1105, 2735, and 4115 Hz;  
% /i/: 437, 2761, 3372, and 4352 Hz

%%
% Assuming your struct is named 'dataStruct' and it has a field 'animal'
animalListSelected = [943, 963, 832];
animalValues = [data.animal];
fieldValues = [data.field];
isAnimalSelected = ismember(animalValues, animalListSelected);
numSelectedAnimals = sum(isAnimalSelected);
% Now count how many selected animals belong to each field (1, 2, 3, or 4)
numField1 = sum(isAnimalSelected & (fieldValues == 1));
numField2 = sum(isAnimalSelected & (fieldValues == 2));
numField3 = sum(isAnimalSelected & (fieldValues == 3));
numField4 = sum(isAnimalSelected & (fieldValues == 4));

% Display the results
fprintf('Number of matching animals: %d\n', numSelectedAnimals);

fprintf('Number of matching animals in field 1: %d\n', numField1);
fprintf('Number of matching animals in field 2: %d\n', numField2);
fprintf('Number of matching animals in field 3: %d\n', numField3);
fprintf('Number of matching animals in field 4: %d\n', numField4);





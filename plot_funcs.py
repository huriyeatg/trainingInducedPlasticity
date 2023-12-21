# This code  has plotting functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import statsmodels.api as sm
from scipy import stats
from scipy.io import loadmat
import statsmodels.formula.api as smf


import os


def set_figure(size='double'):
    from matplotlib import rcParams
        # set the plotting values
    if size == 'single':
        rcParams['figure.figsize'] = [3.5, 7.2]
    elif size == 'double':
        rcParams['figure.figsize'] = [7.2, 7.2]


    rcParams['font.size'] = 10
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    rcParams['axes.spines.right']  = False
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.left']   = True
    rcParams['axes.spines.bottom'] = True

    params = {'axes.labelsize': 'large',
            'axes.titlesize':'large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large',
            'legend.fontsize': 'large'}
    
    rcParams.update(params)

def save_figure(name, base_path):
    plt.savefig(os.path.join(base_path, f'{name}.png'), 
                bbox_inches='tight', transparent=False, dpi = 300)
    plt.savefig(os.path.join(base_path, f'{name}.svg'), 
               bbox_inches='tight', transparent=True)

def plotSSAacrossfields (eng, range_name, ax):
    # Get data from Matlab structure in numpy arrays
    timbre_data = np.array(eng.prepareSSAData('Timbre', nargout=1))
    pitch_data = np.array(eng.prepareSSAData('Pitch', nargout=1))
    control_data = np.array(eng.prepareSSAData('Control', nargout=1))

    range_names = ['Timbre', 'F0', 'Space', 'Space-F0', 'Space-Timbre', 'F0-Timbre']
    index = range_names.index(range_name)
    
    # Define the fields and their corresponding codes in column 7 of the data
    fields = {1: 'A1', 2: 'AAF', 3: 'PPF', 4: 'PSF'}
    colors =[(0.3, 0.3, 0.3), (0.8, 0, 1), (0, 0.4, 1)]

    box_data = []

    # Prepare data for boxplot for each field
    for j, (field_code, field_name) in enumerate(fields.items(), start=1):
        # Check if there is data for the given field in the Control dataset
        if np.any(control_data[:, 6] == field_code):
            df = pd.DataFrame()
            df['Value'] = control_data[control_data[:, 6] == field_code, index]
            df['Training Group'] = 'Control'
            df['Field'] = field_name
            box_data.append(df)

        # Check if there is data for the given field in the Timbre dataset
        if np.any(timbre_data[:, 6] == field_code):
            df = pd.DataFrame()
            df['Value'] = timbre_data[timbre_data[:, 6] == field_code, index]
            df['Training Group'] = 'T 2AFC'
            df['Field'] = field_name
            box_data.append(df)

        # Check if there is data for the given field in the Pitch dataset
        if np.any(pitch_data[:, 6] == field_code):
            df = pd.DataFrame()
            df['Value'] = pitch_data[pitch_data[:, 6] == field_code, index ]
            df['Training Group'] = 'T/P GNG'
            df['Field'] = field_name
            box_data.append(df)

        # Concatenate all dataframes
        all_data = pd.concat(box_data)

    # Plot the boxplot using seaborn
    #sns.boxplot(x='Field', y='Value', hue='Training Group', data=all_data, palette=colors, ax=ax, showfliers=False )
    sns.swarmplot(x='Field', y='Value', hue='Training Group', data=all_data, palette=colors, ax=ax, size = 3, dodge=True)

    # Set the title for each subplot
    #ax.set_title(range_name)
    ax.set_ylabel(range_name +'\n(% Variance explained)')
    ax.set_xlabel('Cortical Field')
    ax.legend(frameon=False)
    
def plotBehaviorTimbre (axisAll):
    dataVowel = [ 85,90,87] # This data is from mean across all stim in this dataset: r'C:\Users\Huriye\Documents\code\trainingInducedPlasticity\info_data\behData_change detection.mat'
    ax = axisAll[0]
    ax.plot(dataVowel, 's', color='m', markersize=10)
    ax.axhline(50, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([1,2,3])
    ax.set_ylim(20,100)
    ax.set_xlim(-0.5,2.5)
    ax.text(-0.4,21, 'n = 3', verticalalignment='bottom', horizontalalignment='left')

    ax = axisAll[1]
    sub1 = [84,52,46]
    ax.plot(sub1, 'o', color='b', markersize=10)
    sub1 = [76,48,66]
    ax.plot(sub1, 'v', color='b', markersize=10,)
    ax.axhline(25, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['/i/','/u/','/$\epsilon$/'])
    ax.set_ylim(20,100)
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel('Vowel identification\n(% Correct)')
    ax.set_xlabel('Target Vowel')
    ax.text(-0.4, 26, 'n = 2', verticalalignment='bottom', horizontalalignment='left')

def plotBehaviorPitch (axisAll):
    # Load .mat file
    ax = axisAll[0]
    beh_data_path = r'C:\Users\Huriye\Documents\code\trainingInducedPlasticity\info_data\behData_change detection.mat'
    mat = loadmat(beh_data_path)
    df = pd.DataFrame(mat['score'].T*100, columns=['Subject 1', 'Subject 2', 'Subject 3'])
    df['stim'] = mat['stim'][1,:-3].T
    df['mean_subjects'] = df[['Subject 1', 'Subject 2', 'Subject 3']].mean(axis=1)

    sns.lineplot(x='stim', y='mean_subjects', data=df, color = 'magenta', linewidth=2.5, ax = ax)
    sns.scatterplot(x='stim', y='Subject 1', data=df, markers = 's',  color="magenta", size = 20, ax = ax)
    sns.scatterplot(x='stim', y='Subject 2', data=df, markers = 's', color="magenta",size = 20, ax = ax)
    sns.scatterplot(x='stim', y='Subject 3', data=df, markers = 's', color="magenta",size = 20, ax = ax)
    ax.set_ylim([20, 100])
    ax.axhline(50, color='grey', linestyle='--')
    ax.set_ylabel('Vowel identification\n(% Correct)')
    ax.set_xlabel('F0 (Hz)')
    ax.legend_.remove()
    ax.text(140,21, 'n = 3', verticalalignment='bottom', horizontalalignment='left')
    

    ax = axisAll[1]
    sub1 = [60,78,80]
    ax.plot(sub1, 'o', color='b', markersize=10)
    sub1 = [50,75,76]
    ax.plot(sub1, 'v', color='b', markersize=10,)
    ax.axhline(25, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['336','556', '951'])
    ax.set_ylim(20,100)
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel('F0 change detection\n(% Correct)')
    ax.set_xlabel('F0 (Hz)')
    ax.text(-0.4, 26, 'n = 2', verticalalignment='bottom', horizontalalignment='left')

def plotVowelSamples (axisAll):
    vowel_data_F = [(1551, 936), (2058, 730), (1105, 460), (2761, 437)]
    vowel_ids = ['a', '$\epsilon$', 'u', 'i']
    colors_F = ['k', 'm', 'm', 'b']
    shapes_F = ['s', 'o', 'o', 'd']  # 's' for square, 'o' for circle

    # vowel plot
    ax = axisAll[0]
    for (y,x), color, shape, vid in zip(vowel_data_F, colors_F, shapes_F, vowel_ids):
        ax.scatter(x, y, s=150, marker=shape, c=color, edgecolors='none')
        ax.text(x+100, y+100, F'/{vid}/', fontsize=12, ha='center', va='center')

    ax.set_xlim(100, 1200)
    ax.set_xticks(range(300, 1201, 300))
    ax.set_ylim(800, 3200)
    ax.set_yticks(range(1000, 3001, 1000))
    ax.set_yticklabels(['1', '2', '3'])
    ax.set_xlabel('F1 (Hz)')
    ax.set_ylabel('F2 (kHz)')
    # Differences 
    ax = axisAll[1]

    F2_values = [1551, 2058, 1105, 2761]
    F1_values = [936, 730, 460, 437]
    colors = [  (0.819115724721261, 0.819115724721261, 0.819115724721261), # even more light grey
                (0.9295040369088812, 0.9295040369088812, 0.9295040369088812), #done
                (0.35912341407151094, 0.35912341407151094, 0.35912341407151094),# grey
                (0.5085736255286428, 0.5085736255286428, 0.5085736255286428), # light grey
                (0.6770011534025375, 0.6770011534025375, 0.6770011534025375),#  more light grey  
                (0.1679354094579008, 0.1679354094579008, 0.1679354094579008), # black -Done
                ] #
    colorInd = 0
    for i in range(len(F1_values)):
        for j in range(i + 1, len(F1_values)):
            delta_F1 = abs(F1_values[i] - F1_values[j])
            delta_F2 = abs(F2_values[i] - F2_values[j])
            ax.scatter(delta_F1, delta_F2, s=100, color= colors[colorInd], marker='s')
            ax.text(delta_F1+90, delta_F2, f"/{vowel_ids[i]}-{vowel_ids[j]}/", fontsize=12, ha='center', va='center')
            colorInd += 1
    ax.set_xlim(-50, 650)
    ax.set_xticks(range(200, 601, 200))
    ax.set_ylim(100, 2000)
    ax.set_yticks(range(500, 2001, 500))
    ax.set_yticklabels(['0.5','1','1.5','2'])
    ax.set_xlabel('$\Delta$ F1 (Hz)')
    ax.set_ylabel('$\Delta$ F2 (kHz)')

def plotVowelSSA(eng,feature, ax):
    
    ax = ax[0]
    # Get data from Matlab structure in numpy arrays
    timbre_data  = np.array(eng.prepareTimbreData('Timbre',feature, nargout=1))
    pitch_data   = np.array(eng.prepareTimbreData('Pitch',feature, nargout=1))
    control_data = np.array(eng.prepareTimbreData('Control',feature, nargout=1))

    # Define the fields and their corresponding codes in column 7 of the data
    vowels = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}

    # We will collect the box plot data and positions here
    box_data = []

    # Prepare data for boxplot for each field
    for j, (field_code, vowel) in enumerate(vowels.items()):
        # Control dataset
        df = pd.DataFrame()
        df['Value'] = control_data[:,j]#[control_data[:, 6] == field_code, i]
        df['Training Group'] = 'Control'
        df['Field'] = vowel
        box_data.append(df)

        #  Timbre trained dataset
        df = pd.DataFrame()
        df['Value'] = timbre_data[:,j]#[timbre_data[:, 6] == field_code, i]
        df['Training Group'] = 'T 2AFC'
        df['Field'] = vowel
        box_data.append(df)

        # Pitch trained dataset
        df = pd.DataFrame()
        df['Value'] = pitch_data[:,j]#[pitch_data[:, 6] == field_code, i]
        df['Training Group'] = 'T/P GNG'
        df['Field'] = vowel
        box_data.append(df)

    # Concatenate all dataframes
    all_data = pd.concat(box_data)
    all_data['Training Group_Field'] = all_data['Training Group'] + '_' + all_data['Field']

    color_palettes = {
    'Control': sns.color_palette("Greys", len(vowels))[::-1],
    'T 2AFC': sns.light_palette('magenta', n_colors=len(vowels), reverse=True),
    'T/P GNG': sns.light_palette('b', n_colors=len(vowels), reverse=True)}

    unique_fields = all_data['Field'].unique()

    all_data['Color'] = all_data.apply(lambda row: color_palettes[row['Training Group']][np.where(unique_fields == row['Field'])[0][0]], axis=1)

    # Create a dictionary that maps 'Type_Field' to 'Color'
    color_dict = dict(zip(all_data['Training Group_Field'], all_data['Color']))

    # Plot the boxplot using seaborn
    sns.boxplot(x='Training Group', y='Value', hue='Training Group_Field', data=all_data, palette=color_dict, ax=ax, showfliers=False )
    #sns.swarmplot(x='Type', y='Value', hue='Type_Field', data=all_data, palette=color_dict, ax=ax, size = 3, dodge=True)

    #handles, labels = ax.get_legend_handles_labels()
    #order = list(range(0, 18, 3)) + list(range(1, 18, 3)) + list(range(2, 18, 3))
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=3, frameon=False)
    ax.legend_.remove()
    # Set the y-axis label
    ax.set_ylabel('Timbre\n(% Variance explained)')
    #ax.set_xticklabels(['/a-u/','/a-e/','/e-i/','/e-u/','/i-a/','/i-u/'])

def plotVowelSSAacrossfields(eng,feature,field_name, ax):
    
    if field_name == 'A1':
        field_id = 1
    elif field_name == 'AAF':
        field_id = 2
    elif field_name == 'PPF':
        field_id = 3
    elif field_name == 'PSF':
        field_id = 4

    # Get data from Matlab structure in numpy arrays
    timbre_data  = np.array(eng.prepareTimbreData('Timbre',feature, nargout=1))
    pitch_data   = np.array(eng.prepareTimbreData('Pitch',feature, nargout=1))
    control_data = np.array(eng.prepareTimbreData('Control',feature, nargout=1))

    # Define the fields and their corresponding codes in column 7 of the data
    vowels = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}

    # We will collect the box plot data and positions here
    box_data = []

    # Prepare data for boxplot for each field
    for j, (vowel_code, vowel) in enumerate(vowels.items()):
        # Control dataset
        df = pd.DataFrame()
        df['Value'] = control_data[control_data[:, 6] == field_id, j]
        df['Training Group'] = 'Control'
        df['Field'] = vowel
        box_data.append(df)

        #  Timbre trained dataset
        df = pd.DataFrame()
        df['Value'] = timbre_data[timbre_data[:, 6] == field_id, j]
        df['Training Group'] = 'T 2AFC'
        df['Field'] = vowel
        box_data.append(df)

        # Pitch trained dataset
        df = pd.DataFrame()
        df['Value'] = pitch_data[pitch_data[:, 6] == field_id, j]
        df['Training Group'] = 'T/P GNG'
        df['Field'] = vowel
        box_data.append(df)

    # Concatenate all dataframes
    all_data = pd.concat(box_data)
    all_data['Training Group_Field'] = all_data['Training Group'] + '_' + all_data['Field']

    color_palettes = {
    'Control': sns.color_palette("Greys", len(vowels))[::-1],
    'T 2AFC': sns.light_palette('magenta', n_colors=len(vowels), reverse=True),
    'T/P GNG': sns.light_palette('b', n_colors=len(vowels), reverse=True)}

    unique_fields = all_data['Field'].unique()

    all_data['Color'] = all_data.apply(lambda row: color_palettes[row['Training Group']][np.where(unique_fields == row['Field'])[0][0]], axis=1)

    # Create a dictionary that maps 'Type_Field' to 'Color'
    color_dict = dict(zip(all_data['Training Group_Field'], all_data['Color']))

    # Plot the boxplot using seaborn
    #sns.swarmplot(x='Field', y='Value', hue='Type', data=all_data, palette=color_dict, ax=ax, size = 3, dodge=True)
    sns.boxplot(x='Training Group', y='Value', hue='Training Group_Field', data=all_data, palette=color_dict, ax=ax, showfliers=False )
    ax.legend_.remove()
    
    # Set the y-axis label
    ax.set_ylabel(field_name)
    ax.set_ylim(-5,60)
  
    # Turn off right and top spines (the lines marking the axes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Turn off ticks
    ax.yaxis.set_ticks_position('none') 
    ax.xaxis.set_ticks_position('none')

    # Set y-label visibility
    ax.yaxis.label.set_visible(True)
    ax.xaxis.set_visible(True)

    #ax.set_xticklabels(['/a-u/','/a-e/','/e-i/','/e-u/','/i-a/','/i-u/'])

def generateDataValuesForGLMM_timbre(eng):
    field_nameList = {1: 'A1', 2: 'AAF', 3: 'PSF', 4: 'PPF'}
    group_types = {1:'Control', 2:'Timbre', 3:'Pitch'}
    feature = 'Timbre'

    # Get data from Matlab structure in numpy arrays
    timbre_data  = np.array(eng.prepareTimbreData('Timbre',feature, nargout=1))
    pitch_data   = np.array(eng.prepareTimbreData('Pitch',feature, nargout=1))
    control_data = np.array(eng.prepareTimbreData('Control',feature, nargout=1))

    # Define the fields and their corresponding codes in column 7 of the data
    vowels = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}
    vowel_data_F = [(1551, 936), (2058, 730), (1105, 460), (2761, 437)] # /a/, /e/, /u/, /i/ (F1, F2) for each vowel
    pairs = [('U', 'I'), ('A', 'I'), ('E', 'U'), ('E', 'I'), ('A', 'E'), ('A', 'U')]
    vowel_to_data = {'U': vowel_data_F[2], 'I': vowel_data_F[3], 'A': vowel_data_F[0], 'E': vowel_data_F[1]}
    delta_F = {pair: (abs(vowel_to_data[pair[0]][0] - vowel_to_data[pair[1]][0]), 
                  abs(vowel_to_data[pair[0]][1] - vowel_to_data[pair[1]][1])) for pair in pairs}
    #print(delta_F[pairs[0]])

    # We will collect the box plot data and positions here
    all_data = []

        # Prepare data for boxplot for each field
    for ind,(field_id, field_name) in enumerate(field_nameList.items()):
        for j, (vowel_code, vowel) in enumerate(vowels.items()):
            for k, (group_code, group_type) in enumerate(group_types.items()):
                if group_type == 'Timbre':
                    current_data = timbre_data[timbre_data[:, 6] == field_id, j]
                elif group_type == 'Pitch': 
                    current_data = pitch_data[pitch_data[:, 6] ==field_id, j]
                elif group_type == 'Control':
                    current_data = control_data[control_data[:, 6] == field_id, j]
                
                for ii,c_dd  in enumerate(current_data):
                    if not np.isnan(c_dd):
                        row = {'TrainingGroup': group_type, 
                            'Unit' : ii + (group_code*1000), # Since all cells are from different animals, adding 1000 is to make sure they are different
                            'Field': field_name,
                            'VowelPair': vowel,
                            'F1' : delta_F[pairs[vowel_code-1]][0],
                            'F2' : delta_F[pairs[vowel_code-1]][1],
                            'Value': c_dd,
                            }
                        all_data.append(row)

    # Concatenate all dataframes
    df = pd.DataFrame(all_data)
    return df

def plotCoefForGLMM_timbre ( eng, savefigpath):
    all_data = generateDataValuesForGLMM_timbre(eng)
    #formula = "Value ~ C(TrainingGroup)*C(VowelPair)*C(Field)"
    formula = "Value ~ C(TrainingGroup) * Field *F1 *F2"
    mixed_effects_model = smf.mixedlm(formula, all_data, groups=all_data['Unit'])

    # Fit the model
    mixed_effects_result = mixed_effects_model.fit()

    print(mixed_effects_result.summary())
    summary = mixed_effects_result.summary()
    summary_html = summary.as_html()

    # Write the HTML string to a file
    saveName = savefigpath + '\\Timbre-GLMM-summary.html'
    with open(saveName, 'w') as f:
        f.write(summary_html)

def plotValidationForMixedEffectModel (eng, ax):
    df = generateDataValuesForGLMM_timbre(eng)
        # Fit the full model
    full_model = smf.mixedlm("Value ~ C(TrainingGroup) * Field *F1 *F2", 
                            df, groups=df["Unit"]).fit(reml=False)
    # Fit reduced models with different combinations of variables and interactions
    reduced_model_1 = smf.mixedlm("Value ~ C(TrainingGroup) * Field", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_2 = smf.mixedlm("Value ~ C(TrainingGroup) * Field *F1", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_3 = smf.mixedlm("Value ~ C(TrainingGroup) * Field *F2", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_4 = smf.mixedlm("Value ~ C(TrainingGroup) * Field +F1 + F2", 
                                df, groups=df["Unit"]).fit(reml=False)

    # Collect AIC for each model
    aic_values = [full_model.aic, reduced_model_1.aic, reduced_model_2.aic, 
                reduced_model_3.aic, reduced_model_4.aic]
    aic_values = aic_values - full_model.aic
    model_names = ['Full Model', 'Full Model without F1 & F2 ', 'Full Model ithout F2',
                 'Full Model without F1','Full Model without Formants Interactions']

    # Plot AIC values
    ax.barh(model_names, aic_values, color='k')
    ax.set_title('Model validation across different models ')
    ax.set_xlabel('AIC values reference to Full Model')
def plotSRNormalisation (eng, featureType, axisAll):
    # Define the feature types and training groups
    if featureType == 'Space': # Space 0, F0 1, Timbre 2
        featureInd = 0
    elif featureType == 'F0':
        featureInd = 1
    elif featureType == 'Timbre':
        featureInd = 2
    trainingGroups = ['Control', 'Timbre' ,'Pitch'] # label is different as Matlab/old code uses
    # these labels for training groups
    trainingGroups_title = ['Control', 'T 2AFC' ,'T/P GNG']

    # Create a figure and axis for plotting
    markers = { 'Control': 's-', 'Timbre': 'o-', 'Pitch': 'd-',}
    colors = {'Control': 'gray', 'Timbre': 'magenta', 'Pitch': 'blue'}

    for i, group_type in enumerate(trainingGroups):
        ax = axisAll[i] 
        normalised = np.array(eng.extractData(group_type, 'resp'))
        matStim = np.array(eng.extractData(group_type, 'stim'))

        # Get unique subFeatures from matStim
        subFeatures = np.unique(matStim[:,featureInd])
        df = np.empty((0,0))
        for stim in subFeatures:
            # Extract the normalized data for the current stimulus
            current_data = normalised[matStim[:,featureInd] == stim].flatten()

            if df.size == 0:
                df = np.atleast_2d(current_data)
            else:
                # If the sizes don't match, either truncate or pad current_data
                if df.shape[1] > current_data.shape[0]:  # if df has more columns
                    # Pad current_data with zeros
                    current_data = np.pad(current_data, (0, df.shape[1] - current_data.shape[0]), mode='constant', constant_values = np.nan)
                elif df.shape[1] < current_data.shape[0]:  # if current_data has more columns
                    # Truncate current_data
                    current_data = current_data[:df.shape[1]]
                
                df = np.vstack((df, current_data))
            
        new_df = df# np.zeros(df.shape)  # This creates an empty array of the same shape as df
        #for index in range(df.shape[1]):
        #    new_df[:, index] = (df[:, index] - np.nanmean(df[:, index])) / (np.nanstd(df[:, index])/ np.sqrt(df[:, index].shape[0]))

        means = np.nanmean(new_df, axis=1)
        errors = np.nanstd(new_df, axis=1) / np.sqrt(df.shape[1])

        # Plot the data with error bars
        ax.errorbar(subFeatures, means, yerr=errors, fmt=markers[group_type], color=colors[group_type], label=group_type if stim == -45 else "")
        ax.axvline(0, color='grey', linestyle='--') 
        #Set the x-axis and y-axis labels
        ax.set_xlabel('Location (degree)')
        ax.set_xlim([-60, 60])
        ax.set_xticks([-45, -15, 15, 45])
        ax.set_ylabel(' Spike rate (Hz)')
        ax.set_title(trainingGroups_title[i])

def generateDataForGLMM(eng, featureType):

    if featureType == 'Space': # Space 0, F0 1, Timbre 2
        featureInd = 0
    elif featureType == 'F0':
        featureInd = 1
    elif featureType == 'Timbre':
        featureInd = 2
        
    trainingGroups = ['Control', 'Timbre' ,'Pitch'] # label is different as Matlab/old code uses
    fieldsName = ['A1', 'AAF', 'PPF', 'PSF']
    # these labels for training groups
    df = pd.DataFrame(columns=['trainingGroup','Unit', 'Field', 'Location', 'spike_rate'])
    data = []
    for i, group_type in enumerate(trainingGroups):
        spikeData = eng.extractData(group_type, 'resp', 'True')
        matStim   = eng.extractData(group_type, 'stim', 'True')
        field = np.array(eng.extractData(group_type, 'field'))
        animal = np.array(eng.extractData(group_type, 'animal'))

        for ii, field_type in enumerate(field):
            spikeData_cell = np.array(spikeData[ii])
            matStim_cell   = np.array(matStim[ii]) 
            if matStim_cell.size != 0 and field_type[0] <5:
                # Get unique subFeatures from matSti
                subFeatures = np.unique(matStim_cell[:,featureInd])     
                for stim in subFeatures:
                    # Extract the normalized data for the current stimulus
                    current_data = np.nanmean(spikeData_cell[matStim_cell[:,featureInd] == stim])
                    if not np.isnan(current_data):
                        row = {'TrainingGroup': group_type, 
                            'Unit' : ii +(i*1000), # Since all cells are from different animals, adding 1000 is to make sure they are different
                            'Field': fieldsName[int(field_type[0])-1],
                            'Location': stim, 
                            'spike_rate': current_data}
                        data.append(row)
               
    df = pd.DataFrame(data)
    #df['Field'] = df['Field'].astype('category')
    #df['TrainingGroup'] = df['TrainingGroup'].astype('category')
    #df['TrainingGroup'].cat.reorder_categories(['Control', 'Timbre', 'Pitch'], ordered=True)
    #df['TrainingGroup'].cat.set_categories(['Control', 'Timbre', 'Pitch'], ordered=True)
    df['Location'] = df['Location'].astype('category')
    df['Location'].cat.reorder_categories([-45, -15, 15, 45], ordered=True)
    df['Location'].cat.set_categories([-45, -15, 15, 45], ordered=True)
    return df

def plotCoefForGLMM ( eng, featureType, ax, savefigpath):
    df = generateDataForGLMM(eng, featureType)
    # Define the GLM model with unit as a random effect
    md = smf.mixedlm("spike_rate ~ C(Location, Treatment(-45)) * TrainingGroup + Field", 
                    df, groups=df["Unit"])
    mdf = md.fit()
    print(mdf.summary())
    # Convert the summary result to HTML
    summary = mdf.summary()
    summary_html = summary.as_html()

    # Write the HTML string to a file
    saveName = savefigpath + '\\' + featureType + 'GLMM-model-summary.html'
    with open(saveName, 'w') as f:
        f.write(summary_html)

    coef_df = mdf.summary().tables[1]
    coef_df['var'] = coef_df.index
    coef_df = coef_df.reset_index(drop=True)
    coef_df['Coef.'] = pd.to_numeric(coef_df['Coef.'])
    coef_df_filtered = coef_df[coef_df['var'].str.startswith('C(Location, Treatment(-45))')]

    # Create the coefficient plot
    sns.pointplot(x="Coef.", y="var", data=coef_df_filtered, 
              color='k', join=False, ax=ax, 
              capsize=0.2, # size of the error bar cap
              errwidth=10,  # thickness of error bar line
              ci=68,     # this will draw error bars for standard deviation
              markers='d')
    ax.axvline(0, color='grey', linestyle='--')  
    ax.set_xlabel("Coefficient")
    ax.set_ylabel('')
    ax.set_title(" GLMM coefficient values \nfor 'Location' categories")
    
def plotValidationForGLMM (eng, featureType, ax):
    df = generateDataForGLMM(eng, featureType)
        # Fit the full model
    full_model = smf.mixedlm("spike_rate ~ C(Location, Treatment(-45)) * TrainingGroup + Field", 
                            df, groups=df["Unit"]).fit(reml=False)
    # Fit reduced models with different combinations of variables and interactions
    reduced_model_1 = smf.mixedlm("spike_rate ~ C(Location) * TrainingGroup", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_2 = smf.mixedlm("spike_rate ~ C(Location) * Field", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_3 = smf.mixedlm("spike_rate ~ C(Location) + TrainingGroup + Field", 
                                df, groups=df["Unit"]).fit(reml=False)
    reduced_model_4 = smf.mixedlm("spike_rate ~ C(Location) * Field + TrainingGroup", 
                                df, groups=df["Unit"]).fit(reml=False)

    # Collect AIC for each model
    aic_values = [full_model.aic, reduced_model_1.aic, reduced_model_2.aic, 
                reduced_model_3.aic, reduced_model_4.aic]
    aic_values = aic_values - full_model.aic
    model_names = ['Full Model', 'Full Model Without Field ', 'Full Model Without Training Group',
                 'Full Model Without Interactions','Full Model With Only Field Group Interactions']

    # Plot AIC values
    ax.barh(model_names, aic_values, color='k')
    ax.xlabel('AIC')
    ax.title('AIC values reference to Full Model ')
    ax.xticks(rotation=45)
    ax.show()
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
from pymer4.models import Lmer 
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2
import re
import os
import itertools
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, conversion
from rpy2.robjects.packages import importr
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable



def set_figure(size='double'):
    from matplotlib import rcParams
        # set the plotting values
    if size == 'single':
        rcParams['figure.figsize'] = [3.5, 7.2]
    elif size == 'double':
        rcParams['figure.figsize'] = [7.2, 7.2]


    rcParams['font.size'] = 12
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

def generateSSAdata (eng, range_name):
    timbre_data = np.array(eng.prepareSSAData('Timbre', nargout=1))
    pitch_data = np.array(eng.prepareSSAData('Pitch', nargout=1))
    control_data = np.array(eng.prepareSSAData('Control', nargout=1))

    range_names = ['Timbre', 'F0', 'Location', 'Location-F0', 'Location-Timbre', 'F0-Timbre']
    index = range_names.index(range_name)

    # Define the fields and their corresponding codes in column 7 of the data
    fields = {1: 'A1', 2: 'AAF', 3: 'PPF', 4: 'PSF'}
    group_types = {1:'Control', 2:'T - Id', 3:'TP - Disc'}

    # Prepare data for boxplot for each field
    all_data = []
    for ind,(field_id, field_name) in enumerate(fields.items()):
        for k, (group_code, group_type) in enumerate(group_types.items()):
            if group_type == 'T - Id':
                current_data = timbre_data[timbre_data[:, 6] == field_id, index]
                penetrations = timbre_data[timbre_data[:, 6] == field_id, 7]
            elif group_type == 'TP - Disc': 
                current_data = pitch_data[pitch_data[:, 6] ==field_id, index]
                penetrations = pitch_data[pitch_data[:, 6] == field_id, 7]
            elif group_type == 'Control':
                current_data = control_data[control_data[:, 6] == field_id, index]
                penetrations = control_data[control_data[:, 6] == field_id, 7]
            
            for ii,c_dd  in enumerate(current_data):
                    row = {'TrainingGroup': group_type, 
                        'Unit' : ii + (group_code*1000), # Since all cells are from different animals, adding 1000 is to make sure they are different
                        'Field': field_name,
                        'Value': c_dd,
                        'Penetration':penetrations[ii],
                        }
                    all_data.append(row)
    df = pd.DataFrame(all_data)
    return df

def plotSSAacrossfields (eng, range_name, ax):
    # Get data from Matlab structure in numpy arrays
    all_data = generateSSAdata(eng, range_name)
    
    # Plot the boxplot using seaborn
    colors =[(0.3, 0.3, 0.3), (0.8, 0, 1), (0, 0.4, 1)]
    #sns.boxplot(x='Field', y='Value', hue='Training Group', data=all_data, palette=colors, ax=ax, showfliers=False )
    sns.swarmplot(x='Field', y='Value', hue='TrainingGroup', data=all_data, palette=colors, ax=ax, size = 2, dodge = True)

    # Set the title for each subplot
    #ax.set_title(range_name)
    ax.set_ylabel(range_name +'\n(% Variance explained)')
    ax.set_xlabel('Cortical Field')
    #legend to right top sid
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    

def crossvalidata (df,model_formula ):

    # Enable automatic conversion from pandas to R
    pandas2ri.activate()

    # Import R packages
    lme4 = importr('lme4')
    stats = importr('stats')
    
    # Define the number of folds for cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize a list to store the evaluation metric (e.g., RMSE) for each fold
    eval_metrics = []

    for train_index, test_index in kf.split(df):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        
        # To make sure there is no new data in test set...
        for col in ['Unit', 'Penetration']:
            test_data = test_data[test_data[col].isin(train_data[col])]
        # Convert pandas DataFrames to R DataFrames
        r_train_data = pandas2ri.py2rpy(train_data)
        r_test_data = pandas2ri.py2rpy(test_data)

        # Fit the model on the training set using lmer
        md = lme4.lmer(model_formula, data=r_train_data)

        # Make predictions on the testing set
        y_pred = stats.predict(md, newdata=r_test_data)

        # Evaluate the model using a relevant metric (e.g., RMSE)
        rmse = np.sqrt(mean_squared_error(test_data["Value"], conversion.rpy2py(y_pred)))
        eval_metrics.append(rmse)

    # Calculate the average evaluation metric across all folds
    avg_eval_metric = np.mean(eval_metrics)
    return avg_eval_metric

def GLMMforSSA (eng, range_name):
    
    df = generateSSAdata(eng, range_name)
    # Jan spotted the penetration random effect on fields - so we need to include it in the model
    # model = smf.glm('Value ~ TrainingGroup * Field', data=df,
    #                family=sm.families.Poisson(), groups=df["Unit"]).fit() #
    
    # Create the mixed effects model with 'Penetration' as a random effect, 02 Feb 2024
        # # Attempt 1 from smf library - This is not correct way to go, as we want to give the 
        # # information about same penetration. Unit_penetration makes them more independentm but units in
        # # the same penetration are not independent
        # df['Unit_Penetration'] = df['Unit'].astype(str) + "_" + df['Penetration'].astype(str)
        # md = smf.mixedlm("Value ~ TrainingGroup * Field", 
        #                 df,family=sm.families.Poisson(), groups=df["Unit_Penetration"])
    
    # Lmer allows us two random effects, so we can use it to include both Unit and Penetration
    # as penetration is nested within unit, we can use it as a random effect
    # Activate R
    pandas2ri.activate()
    # Import R packages
    performance = importr('performance')
    lmerTest = importr('lmerTest')

    # Define the mixed effects model formula
    model_formula = 'Value ~ TrainingGroup * Field + (1|Penetration)'

    # Fit the model using lmerTest
    md = lmerTest.lmer(model_formula, data=df)

    # Print the summary
    #print(robjects.r['summary'](md))

    md_summary = robjects.r['coef'](md)
    fixed_effects = md_summary[0]
    fixed_effect_names = fixed_effects.names[:]

    coef_df = pd.DataFrame(robjects.r['summary'](md).rx2('coefficients'))
    coef_df.columns = ['Estimate', 'Std. Error', 'df', 't value', 'Pr(>|t|)']
    coef_df.index = fixed_effect_names
    print(coef_df)

    # Calculate R-squared values in R
    r2_values = performance.r2(md)

    # Extract R-squared values
    marginal_r2 = r2_values[0]
    conditional_r2 = r2_values[1]

    print("AIC", marginal_r2)
    print("Marginal R-squared:", marginal_r2)
    print("Conditional R-squared:", conditional_r2)

    crossValidateValue = crossvalidata (df,model_formula )
    print(f"Cross-validated RMSE: {crossValidateValue:.2f}")
    # Perform post-hoc comparison using Tukey's HSD
    posthoc = pairwise_tukeyhsd(df['Value'], df['TrainingGroup'])
    print(posthoc)
    # Perform post-hoc comparison using Tukey's HSD
    posthoc = pairwise_tukeyhsd(df['Value'], df['Field'])
    print(posthoc)

    # Perform post-hoc comparison for Field within each Training Group
    fields = df['Field'].unique()

    for field in fields:
        subset = df[df['Field'] == field]
        posthoc = pairwise_tukeyhsd(subset['Value'], subset['TrainingGroup'])
        print(f"Post-hoc comparison for Field within {field}:\n{posthoc}\n")
        
def plotBehaviorTimbre (axisAll):
    dataVowel = [ 85,90,87] # This data is from mean across all stim in this dataset: r'C:\Users\Huriye\Documents\code\trainingInducedPlasticity\info_data\behData_change detection.mat'
    ax = axisAll[0]
    markers = ['+', 's', '^']  # Circle, Square, Triangle
    # Plot each data point with a different marker
    for i, value in enumerate(dataVowel):
        ax.plot(i, value, marker=markers[i], color='m', markersize=10, markerfacecolor='none')
    ax.axhline(50, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([1,2,3])
    ax.set_ylim(20,100)
    ax.set_ylabel('% Correct')
    ax.set_xlabel( 'Subject')
    ax.set_title('T - Id')
    ax.set_xlim(-0.5,2.5)
    ax.text(-0.4,21, 'n = 3', verticalalignment='bottom', horizontalalignment='left')

    ax = axisAll[1]
    sub1 = [84,52,46]
    ax.plot(sub1, 'o', color='b', markersize=10, markerfacecolor='none')
    sub1 = [76,48,66]
    ax.plot(sub1, 'v', color='b', markersize=10, markerfacecolor='none')
    ax.axhline(25, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['/i/','/u/','/$\epsilon$/'])
    ax.set_ylim(20,100)
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel('% Correct')
    ax.set_title('TP - Disc')
    ax.set_xlabel('Target Vowel')
    ax.text(-0.4, 26, 'n = 2', verticalalignment='bottom', horizontalalignment='left')

def plotBehaviorPitch (axisAll):
    # Load .mat file
    ax = axisAll[0]
    beh_data_path = r'D:\trainingInducedPlasticity\info_data\behData_change detection.mat'
    mat = loadmat(beh_data_path)
    df = pd.DataFrame(mat['score'].T*100, columns=['Subject 1', 'Subject 2', 'Subject 3'])
    df['stim'] = mat['stim'][1,:-3].T
    df['mean_subjects'] = df[['Subject 1', 'Subject 2', 'Subject 3']].mean(axis=1)

    sns.lineplot(x='stim', y='mean_subjects', data=df, color = 'magenta', linewidth=2.5, ax = ax)
    sns.scatterplot(x='stim', y='Subject 1', data=df, marker = '+',  color="magenta", size= 200, ax = ax)
    sns.scatterplot(x='stim', y='Subject 2', data=df, marker = 's', color="magenta",size = 200, ax = ax)
    sns.scatterplot(x='stim', y='Subject 3', data=df, marker = '^', color="magenta",size = 200, ax = ax)
    ax.set_ylim([20, 100])
    ax.axhline(50, color='grey', linestyle='--')
    ax.set_ylabel('% Correct')
    ax.set_xlabel('F0 (Hz)')
    ax.set_title('T-Id')
    ax.legend_.remove()
    ax.text(140,21, 'n = 3', verticalalignment='bottom', horizontalalignment='left')
    

    ax = axisAll[1]
    sub1 = [60,78,80]
    ax.plot(sub1, 'o', color='b', markersize=10, markerfacecolor='none')
    sub1 = [50,75,76]
    ax.plot(sub1, 'v', color='b', markersize=10,markerfacecolor='none')
    ax.axhline(25, color='grey', linestyle='--')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['336','556', '951'])
    ax.set_ylim(20,100)
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel('% Correct')
    ax.set_title('TP-Disc')
    ax.set_xlabel('F0 (Hz)')
    ax.text(-0.4, 26, 'n = 2', verticalalignment='bottom', horizontalalignment='left')

def plotVowelSamples (axisAll):
    vowel_data_F = [(1551, 936), (2058, 730), (1105, 460), (2761, 437)]
    vowel_ids = ['a', '$\epsilon$', 'u', 'i']
    colors_F = ['b', 'm', 'm', 'k']
    shapes_F = ['s', 'o', 'o', 'd']  # 's' for square, 'o' for circle

    # vowel plot
    ax = axisAll[0]
    for (y,x), color, shape, vid in zip(vowel_data_F, colors_F, shapes_F, vowel_ids):
        ax.scatter(x, y, s=150, marker=shape, c=color, edgecolors='none')
        ax.text(x+100, y+100, F'/{vid}/', fontsize=14, ha='center', va='center')

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
            ax.text(delta_F1+ 110, delta_F2, f"/{vowel_ids[i]}-{vowel_ids[j]}/", fontsize=14, ha='center', va='center')
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
    vowels = {1: 'AU', 2: 'AE', 3: 'EI', 4: 'EU', 5: 'AI', 6: 'UI'}
    #real_vowelIDs = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}
    indexOrderForVowels = [5,4,3,2,1,0]
    # We will collect the box plot data and positions here
    box_data = []

    # Prepare data for boxplot for each field
    for j, (field_code, vowel) in enumerate(vowels.items()):
        # Control dataset
        df = pd.DataFrame()
        df['Value'] = control_data[:,indexOrderForVowels[j]]#[control_data[:, 6] == field_code, i]
        df['Training Group'] = 'Control'
        df['VowelPair'] = vowel
        box_data.append(df)

        #  Timbre trained dataset
        df = pd.DataFrame()
        df['Value'] = timbre_data[:,indexOrderForVowels[j]]#[timbre_data[:, 6] == field_code, i]
        df['Training Group'] = 'T - Id'
        df['VowelPair'] = vowel
        box_data.append(df)

        # Pitch trained dataset
        df = pd.DataFrame()
        df['Value'] = pitch_data[:,indexOrderForVowels[j]]
        df['Training Group'] = 'TP - Disc'
        df['VowelPair'] = vowel
        box_data.append(df)

    # Concatenate all dataframes
    all_data = pd.concat(box_data)
    all_data['Training Group_VowelPair'] = all_data['Training Group'] + '_' + all_data['VowelPair']
    color_palettes = {
    'Control': sns.color_palette("Greys", len(vowels)),
    'T - Id': sns.light_palette('magenta', n_colors=len(vowels)),
    'TP - Disc': sns.light_palette('b', n_colors=len(vowels))}

    unique_fields = all_data['VowelPair'].unique()

    all_data['Color'] = all_data.apply(lambda row: color_palettes[row['Training Group']][np.where(unique_fields == row['VowelPair'])[0][0]], axis=1)

    # Create a dictionary that maps 'Type_Field' to 'Color'
    color_dict = dict(zip(all_data['Training Group_VowelPair'], all_data['Color']))

    # Plot the boxplot using seaborn
    sns.boxplot(x='Training Group', y='Value', hue='Training Group_VowelPair', data=all_data, palette=color_dict, ax=ax, showfliers=False )
    #sns.swarmplot(x='Type', y='Value', hue='Type_Field', data=all_data, palette=color_dict, ax=ax, size = 3, dodge=True)

    #handles, labels = ax.get_legend_handles_labels()
    #order = list(range(0, 18, 3)) + list(range(1, 18, 3)) + list(range(2, 18, 3))
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=3, frameon=False)
    ax.legend_.remove()
    # Set the y-axis label
    ax.set_ylabel('Timbre\n(% Variance explained)')
    #ax.set_xticklabels(['/a-u/','/a-e/','/e-i/','/e-u/','/i-a/','/i-u/'])

def plotVowelSSAacrossTrainingGroups(eng,group_name, ax):        
            
    # Get data from Matlab structure in numpy arrays
    ssa_data  = np.array(eng.prepareTimbreData(group_name, 'Timbre', nargout=1))

    # Define the fields and their corresponding codes in column 7 of the data
    vowels = {1: 'AU', 2: 'AE', 3: 'EI', 4: 'EU', 5: 'AI', 6: 'UI'}
    #real_vowelIDs = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}
    indexOrderForVowels = [5,4,3,2,1,0]
    
    fields = {1: 'A1', 2: 'AAF', 3: 'PPF', 4: 'PSF'}

    if group_name == 'Timbre':
        trainingGroups_title = 'T - Id'
    elif group_name == 'Pitch':
            trainingGroups_title = 'TP - Disc'
    elif group_name == 'Control':
            trainingGroups_title = 'Control'
    
    # We will collect the box plot data and positions here
    box_data = []

    # Prepare data for boxplot for each field
    for j, (vowel_code, vowel) in enumerate(vowels.items()):
        for i, (field_code, field) in enumerate(fields.items()):
            # Control dataset
            df = pd.DataFrame()
            df['Value'] = ssa_data[ssa_data[:, 6] == field_code, indexOrderForVowels[j]]
            df['Field'] = field
            df['Vowel'] = vowel
            df['TrainingGroup'] = trainingGroups_title
            box_data.append(df)


    # Concatenate all dataframes
    all_data = pd.concat(box_data)
    all_data['Field_Vowel'] = all_data['Field'] + '_' + all_data['Vowel']

    color_palettes = {
        'Control': sns.color_palette("Greys", len(vowels)),
        'T - Id': sns.light_palette('magenta', len(vowels)),
        'TP - Disc': sns.light_palette('b', len(vowels))}

    unique_vowels= all_data['Vowel'].unique()

    all_data['Color'] = all_data.apply(lambda row: color_palettes[row['TrainingGroup']][np.where(unique_vowels == row['Vowel'])[0][0]], axis=1)
    # Create a dictionary that maps 'Type_Field' to 'Color'
    color_dict = dict(zip(all_data['Field_Vowel'], all_data['Color']))

    # Plot the boxplot using seaborn
    #sns.swarmplot(x='Field', y='Value', hue='Type', data=all_data, palette=color_dict, ax=ax, size = 3, dodge=True)
    sns.boxplot(x='Field', y='Value', hue='Field_Vowel', data=all_data, palette=color_dict, ax=ax, showfliers=False )
    ax.legend_.remove()
    
    # Set the y-axis label
    ax.set_ylabel(trainingGroups_title)
    ax.set_ylim(-5,60)
    ax.set_yticks(range(0, 61, 25))
    ax.set_yticklabels(['0','25','50'])
  
    # Turn off right and top spines (the lines marking the axes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Turn off ticks
    ax.yaxis.set_ticks_position('none') 
    ax.xaxis.set_ticks_position('none')

    # Set y-label visibility
    ax.yaxis.label.set_visible(True)
    ax.xaxis.set_visible(True)

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

def generateDataValuesForGLMM_timbreSSA(eng):
    field_nameList = {1: 'A1', 2: 'AAF', 3: 'PSF', 4: 'PPF'}
    group_types = {1:'Control', 2:'Timbre', 3:'Pitch'}
    feature = 'Timbre'

    # Get data from Matlab structure in numpy arrays
    timbre_data  = np.array(eng.prepareTimbreData('Timbre',feature, nargout=1))
    pitch_data   = np.array(eng.prepareTimbreData('Pitch',feature, nargout=1))
    control_data = np.array(eng.prepareTimbreData('Control',feature, nargout=1))

    # Define the fields and their corresponding codes in column 7 of the data
    vowels = {1: 'UI', 2: 'AI', 3: 'EU', 4: 'EI', 5: 'AE', 6: 'AU'}
    vowel_data_F = [(936, 1551), (730, 2058), (460,1105), (437, 2761)] # /a/, /e/, /u/, /i/ (F1, F2) for each vowel
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
                    penetrations = timbre_data[timbre_data[:, 6] == field_id, 6]
                elif group_type == 'Pitch': 
                    current_data = pitch_data[pitch_data[:, 6] ==field_id, j]
                    penetrations = pitch_data[pitch_data[:, 6] == field_id, 6]
                elif group_type == 'Control':
                    current_data = control_data[control_data[:, 6] == field_id, j]
                    penetrations = control_data[control_data[:, 6] == field_id, 7]

                for ii,c_dd  in enumerate(current_data):
                    if not np.isnan(c_dd):
                        row = {'TrainingGroup': group_type, 
                            'Unit' : ii + (group_code*1000), # Since all cells are from different animals, adding 1000 is to make sure they are different
                            'Field': field_name,
                            'VowelPair': vowel,
                            'F1' : delta_F[pairs[vowel_code-1]][0],
                            'F2' : delta_F[pairs[vowel_code-1]][1],
                            'Value': c_dd,
                            'Penetration':penetrations[ii],
                            }
                        all_data.append(row)

    # Concatenate all dataframes
    df = pd.DataFrame(all_data)
    return df

def generatePredictionDataForGLMM_timbre(eng):
    df = generateDataValuesForGLMM_timbreSSA(eng)
    # Activate R
    pandas2ri.activate()
    # Import R packages
    performance = importr('performance')
    lmerTest = importr('lmerTest')
    #xtable = importr('xtable')

    # Define the mixed effects model formula
    model_formula = 'Value ~ Field + TrainingGroup + F1 + F2 + TrainingGroup:F1 + TrainingGroup:F2 + Field:F1 + Field:F2 + F1:F2 + (1|Unit) + (1|Penetration)'

    # Fit the model using lmerTest
    md = lmerTest.lmer(model_formula, data=df)

    # Generate a range of values for F1 and F2
    f1_values = np.linspace(min(df['F1']), max(df['F1']), num=400)
    f2_values = np.linspace(min(df['F2']), max(df['F2']), num=400)
    fields = df['Field'].unique()
    training_groups = df['TrainingGroup'].unique()

    value =[]
    for field in fields:
        for training_group in training_groups:
            for f1 in f1_values:
                for f2 in f2_values:
                    value.append([field, training_group, f1, f2])

    # Create a new DataFrame with repeated combinations and F1, F2 values
    new_data = pd.DataFrame(value, columns=['Field', 'TrainingGroup', 'F1', 'F2'])

    # Convert pandas dataframe to R dataframe
    new_rdf = pandas2ri.py2rpy(new_data)

    robjects.globalenv['newdata'] = new_rdf
    robjects.globalenv['md'] = md
    robjects.r('newdata$predicted_values <- predict(md, newdata=newdata)')

    # Convert R dataframe back to pandas dataframe
    new_data = pandas2ri.rpy2py(robjects.r['newdata'])

    return new_data

def plotPrediction2d(eng, axisAll):
    new_data = generatePredictionDataForGLMM_timbre(eng)

    colors = {'Control': 'gray', 'Timbre': 'magenta', 'Pitch': 'blue'}
    trainingGroups = ['Control', 'Timbre' ,'Pitch'] 
    trainingGroups_title = ['Control', 'T - Id' ,'TP - Disc']
    # Plot the F1 
    ax = axisAll[0]
    binSize = 4
    for i, training_group in enumerate(trainingGroups):
        # Get the index for each training group
        f1 = new_data['F1'][new_data['TrainingGroup'] == training_group]
        pvalues = new_data['predicted_values'][new_data['TrainingGroup'] == training_group]
        
        # # Define bin edges. Adjust the range and number of bins as needed.
        # bins = np.linspace(min(f1), max(f1), num=len(f1.unique())//binSize)

        # # Create a new column in new_data with the bin each 'F1' value falls into
        # new_data['F1_bins'] = pd.cut(new_data['F1'], bins, labels=False)

        # for k, bin in enumerate(np.unique(new_data['F1_bins'])):
        #     # Filter pvalues based on unique F1_bins
        #     filtered_pvalues = pvalues[new_data['F1_bins'] == bin]

        #     # Calculate the mean of the F1 values in this bin for plotting
        #     f1_bin_mean = new_data.loc[new_data['F1_bins'] == bin, 'F1'].mean()
        #     ax.plot(f1_bin_mean, np.mean(filtered_pvalues), marker='o', color=colors[training_group], label=trainingGroups_title[i])


        for f1_unique in f1.unique():
            # Filter pvalues based on unique f1_values
            filtered_pvalues = pvalues[new_data['F1'] == f1_unique]
            # Define the window size for the moving average
            window_size = 1

            # Calculate the simple moving average
            smoothed_pvalues = filtered_pvalues.rolling(window=window_size, center=True,min_periods=1).mean()
            # Handle NaNs at the edges by using min_periods=1
            # smoothed_pvalues = filtered_pvalues.rolling(window=window_size, center=True, min_periods=1).mean()
            #ax.plot(smoothed_pvalues.index, smoothed_pvalues, color=colors[training_group],
            #            label=trainingGroups_title[i], marker='.', markersize=4)
            ax.plot(f1_unique, np.mean(filtered_pvalues), marker='.', 
                       color=colors[training_group] , label=trainingGroups_title[i],
                     markersize= 4)

    ax.set_ylim ( 0, 22)
    ax.set_xlim(0, 600)
    ax.set_xticks(range(0, 601, 200))
    ax.set_xlabel('$\Delta$ F1 (Hz)')
    ax.set_ylabel('Predicted Change in\nDiscriminability')

    # Plot the F2
    ax = axisAll[1]
    for i, training_group in enumerate(trainingGroups):
        # Get the index for each training group
        f2 = new_data['F2'][new_data['TrainingGroup'] == training_group]
        pvalues = new_data['predicted_values'][new_data['TrainingGroup'] == training_group]
        
        for f2_unique in f2.unique():
            # Filter pvalues based on unique f1_values
            filtered_pvalues = pvalues[new_data['F2'] == f2_unique]
            window_size = 1
            smoothed_pvalues = filtered_pvalues.rolling(window=window_size, center=True,min_periods=1).mean()
 
            ax.plot(f2_unique, np.mean(filtered_pvalues), marker='.', 
                   color=colors[training_group] , label=trainingGroups_title[i],
                   markersize=4)
            #ax.plot(filtered_pvalues.index, np.mean(filtered_pvalues), label='Original', marker='o')
            # ax.plot(smoothed_pvalues.index, smoothed_pvalues,color=colors[training_group],
            #             label=trainingGroups_title[i], marker='.', markersize=4)

    ax.set_ylim ( 0, 22)
    ax.set_xlim(0, 2000)
    ax.set_xticks(range(500, 2001, 500))
    ax.set_xticklabels(['0.5','1','1.5','2'])
    ax.set_xlabel('$\Delta$ F2 (kHz)')
    ax.set_yticklabels([])

def plotPrediction3d(eng, axisAll, coloraxis):
    new_data = generatePredictionDataForGLMM_timbre(eng)

    colors = {'Control': 'gray', 'Timbre': 'magenta', 'Pitch': 'blue'}
    trainingGroups = ['Control', 'Timbre' ,'Pitch'] 
    trainingGroups_title = ['Control', 'T - Id' ,'TP - Disc']
    zmax = [ 30, 20, 10]

    for i, training_group in enumerate(trainingGroups):
        # Filter data for the specific training group
        group_data = new_data[new_data['TrainingGroup'] == training_group]
        
        ax = axisAll[i]
        # Assuming you have a DataFrame 'df' with columns 'F1', 'F2', and 'values' you wish to interpolate
        f1_values = np.linspace(min(group_data['F1']), max(group_data['F1']), num=400)
        f2_values = np.linspace(min(group_data['F2']), max(group_data['F2']), num=400)

        grid_x, grid_y = np.meshgrid(f1_values, f2_values)
        grid_z = griddata((group_data['F1'], group_data['F2']), group_data['predicted_values'], (grid_x, grid_y), method='cubic')
        
        im = ax.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap='viridis')  # Choose a colormap that fits your data
        
        #ax.plot_trisurf(group_data['F1'], group_data['F2'], group_data['predicted_values'],
        #                color=colors[training_group], label=f'Training Group {training_group}')
        ax.set_xlabel('$\Delta$ F1 (Hz)')
        ax.set_title(trainingGroups_title[i])
        #ax.zaxis.set_rotate_label(True)
        ax.tick_params(axis='x', labelsize='medium')
        if i == 0:
            ax.tick_params(axis='y', labelsize='medium')
            ax.set_ylabel('$\Delta$ F2 (kHz)')
            ax.yaxis.labelpad = 10  # Increase the padding between the y-axis label and the plot
        else:
            ax.set_yticklabels([])
            # add color bar 
            cbar = plt.colorbar(im, cax=coloraxis)
            cbar.set_label('Predicted Change\nin Discriminability', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize='medium')
def GLMM_timbreSSA (eng, savefigpath):
    df = generateDataValuesForGLMM_timbreSSA(eng)
    # Activate R
    pandas2ri.activate()
    # Import R packages
    performance = importr('performance')
    lmerTest = importr('lmerTest')
    #xtable = importr('xtable')

    # Define the mixed effects model formula
    model_formula = 'Value ~ Field + TrainingGroup + F1 + F2 + TrainingGroup:F1 + TrainingGroup:F2 + Field:F1 + Field:F2 + F1:F2 + (1|Unit) + (1|Penetration)'

    # Fit the model using lmerTest
    md = lmerTest.lmer(model_formula, data=df)

    # Print the summary
    print(robjects.r['summary'](md))

    # Calculate R-squared values in R
    r2_values = performance.r2(md)

    # Extract R-squared values
    marginal_r2 = r2_values[0]
    conditional_r2 = r2_values[1]

    print("Marginal R-squared:", marginal_r2)
    print("Conditional R-squared:", conditional_r2)

    crossValidateValue = crossvalidata (df,model_formula )
    print(f"Cross-validated RMSE: {crossValidateValue:.2f}")

def plotValidationForMixedEffectModel (eng, ax):
    df = generateDataValuesForGLMM_timbre(eng)
        # Fit the full model
    formula = "Value ~ Field + TrainingGroup + F1 + F2 + TrainingGroup:Field + TrainingGroup:F1 + TrainingGroup:F2 + Field:F1 + Field:F2 + F1:F2 + TrainingGroup:F1:F2:Field"
    full_model = smf.glm(formula, df,family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)
    # Fit reduced models with different combinations of variables and interactions
    formula = "Value ~ Field + TrainingGroup + F1 + F2 + TrainingGroup:Field + TrainingGroup:F1 + TrainingGroup:F2 + Field:F1 + Field:F2 + F1:F2 + Field:F1:F2 +  TrainingGroup:Field:F1 + TrainingGroup:Field:F2 + TrainingGroup:F1:F2 + Field:F1:F2 + TrainingGroup:F1:F2:Field"
    reduced_model_1 = smf.glm(formula, df,family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)
    formula = "Value ~ Field + TrainingGroup + F1 + F2 + TrainingGroup:Field + TrainingGroup:F1 + TrainingGroup:F2 + Field:F1 + Field:F2 + F1:F2"
    reduced_model_2 = smf.glm(formula, df,family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)
    reduced_model_3 = smf.glm("Value ~ C(TrainingGroup) * Field + F1 + F2", 
                                df,family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)
    reduced_model_4 = smf.glm("Value ~ C(TrainingGroup) * Field", 
                                df,family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)

    # Collect AIC for each model
    aic_values = [(1-full_model.deviance/full_model.null_deviance), (1-reduced_model_1.deviance/reduced_model_1.null_deviance),
                   (1-reduced_model_2.deviance/reduced_model_2.null_deviance),
                   (1-reduced_model_3.deviance/reduced_model_3.null_deviance),
                   (1-reduced_model_4.deviance/reduced_model_4.null_deviance)]
    #aic_values = [full_model.aic, reduced_model_1.aic, reduced_model_2.aic, 
     #           reduced_model_3.aic, reduced_model_4.aic]
    #aic_values = aic_values - full_model.aic
    model_names = ['Full Model (no 3 interactions)', 'Full Model with All Interactions ', 'Full Model without 3 & 4 interactions',
                 'Full Model without F1 & F2 interations','Full Model without F1 & F2']

    # Plot AIC values
    ax.barh(model_names, aic_values, color='k')
    ax.set_title('Model validation across different models ')
    ax.set_xlabel('AIC values reference to Full Model')

def plotSRNormalisation (eng, featureType, axisAll):
    # Define the feature types and training groups
    if featureType == 'Location': # Space 0, F0 1, Timbre 2
        featureInd = 0
    elif featureType == 'F0':
        featureInd = 1
    elif featureType == 'Timbre':
        featureInd = 2
    trainingGroups = ['Control', 'Timbre' ,'Pitch'] # label is different as Matlab/old code uses
    # these labels for training groups
    trainingGroups_title = ['Control', 'T - Id' ,'TP - Disc']

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

def generateDataForGLMM_spikeData(eng, featureType):

    if featureType == 'Location': # Location 0, F0 1, Timbre 2
        featureInd = 0
    elif featureType == 'F0':
        featureInd = 1
    elif featureType == 'Timbre':
        featureInd = 2
    
    field_nameList = ['A1','AAF','PSF','PPF']
    group_types = {1:'Control', 2:'Timbre', 3:'Pitch'} # label is different as Matlab/old code uses
    group_names = ['Control', 'T - Id', 'TP - Disc']
    # these labels for training groups
    df = pd.DataFrame(columns=['trainingGroup','Unit', 'Field', 'Location', 'spike_rate'])
    data = []
    for i, (group_code, group_type) in enumerate(group_types.items()):
        spikeData = eng.extractData(group_type, 'resp', 'True')
        matStim   = eng.extractData(group_type, 'stim', 'True')
        fieldStim   = eng.extractData(group_type, 'field', 'False')
        penetration = eng.extractData(group_type,  'shrankUnique', 'False')
        
        for ii in range(len(spikeData)):
            spikeData_cell = np.array(spikeData[ii])
            matStim_cell   = np.array(matStim[ii]) 
            if (matStim_cell.size != 0):
                # Get unique subFeatures from matSti
                subFeatures = np.unique(matStim_cell[:,featureInd])   
                for stim in subFeatures:
                    # Extract the normalized data for the current stimulus
                    current_data = np.nanmean(spikeData_cell[matStim_cell[:,featureInd] == stim])
                    if (not np.isnan(current_data)) & (fieldStim[ii]<5):
                        row = {'TrainingGroup': group_names[i], 
                            'Unit' : ii +(group_code*1000), # Since all cells are from different animals, adding 1000 is to make sure they are different
                            'Field': field_nameList[int(fieldStim[ii])-1],
                            'Location': str(int(stim)), 
                            'spike_rate': current_data,
                            'Penetration':penetration[ii] }
                        data.append(row)
    df = pd.DataFrame(data)
    df['Location'] = df['Location'].astype(str)
    df['Location'] = pd.Categorical(df['Location'], categories=['-45', '-15', '15', '45'], ordered=True)
    #df['Location'] = df['Location'].astype('category')
    #df['Location'].cat.reorder_categories([-45, -15, 15, 45], ordered=True)
    #df['Location'].cat.set_categories([-45, -15, 15, 45], ordered=True)
    return df

def plotCoefForGLMM_spikeData ( eng, featureType, ax, savefigpath):
    # Activate R
    pandas2ri.activate()
    performance = importr('performance')
    lmerTest = importr('lmerTest')

    df = generateDataForGLMM_spikeData(eng, featureType)
    # Define the GLM model with unit as a random effect - This method is not nested
    # df['Unit_Penetration'] = df['Unit'].astype(str) + "_" + df['Penetration'].astype(str)
    # md = smf.mixedlm("spike_rate ~ C(Location, Treatment(-45)) * TrainingGroup * Field", 
    #                 df,groups=df["Unit_Penetration"])
    # 
    model_formula = 'spike_rate ~ Location * TrainingGroup * Field +(1|Unit) + (1|Penetration)'
    #model_formula = 'spike_rate ~ factor(Location, levels=c("-45", "-15", "15", "45")) * TrainingGroup * Field + (1|Unit) + (1|Penetration)'
    #model_formula = 'spike_rate ~ factor(Location, levels=c(-45, -15, 15, 45)) * TrainingGroup * Field + (1|Unit) + (1|Penetration)'

    # Fit the model using lmerTest
    r_df = pandas2ri.py2rpy(df)
    robjects.globalenv['df'] = r_df

    #Set the reference level for 'Location' in R
    robjects.r('''
               df$Location <- factor(df$Location, levels=c("-45", "-15", "15", "45"), ordered = FALSE)
               options(contrasts=c("contr.treatment", "contr.poly"))
               ''')

    md = lmerTest.lmer(model_formula, data=robjects.globalenv['df'])

    # Print the summary
    print(robjects.r['summary'](md))

    # Calculate R-squared values in R
    r2_values = performance.r2(md)

    # Extract R-squared values
    marginal_r2 = r2_values[0]
    conditional_r2 = r2_values[1]

    print("Marginal R-squared:", marginal_r2)
    print("Conditional R-squared:", conditional_r2)

    md_summary = robjects.r['coef'](md)
    fixed_effects = md_summary[0]
    fixed_effect_names = fixed_effects.names[:]

    coef_df = pd.DataFrame(robjects.r['summary'](md).rx2('coefficients'))
    coef_df.columns = ['Estimate', 'Std. Error', 'df', 't value', 'Pr(>|t|)']
    coef_df.index = fixed_effect_names
    

    filter_conditions = ['TrainingGroup', 'Field']
    coef_df_filtered = coef_df[coef_df.index.str.startswith('Location') & coef_df.index.to_series().apply(lambda x: all(word in x for word in filter_conditions))]
    coef_df_filtered['ylabel'] = coef_df_filtered.index.to_series().apply(
    lambda x: ' '.join(filter(None, [item for sublist in re.findall(r'Location(-?\d+)|TrainingGroup\s*[\W_]*\s*((?:T - Id|TP - Disc))|Field([A-Za-z]+)', x) for item in sublist])))

    # Create the coefficient plot
    sns.pointplot(x="Estimate", y="ylabel", data=coef_df_filtered, 
                color='k', join=False, ax=ax, 
                capsize=0.2, # size of the error bar cap
                errwidth=10,  # thickness of error bar line
                ci=68,     # this will draw error bars for standard deviation
                markers='d')
    ax.axvline(0, color='grey', linestyle='--')  
    ax.set_xlabel("Coefficient")
    ax.set_yticklabels(coef_df_filtered['ylabel'])
    ax.set_ylabel('')
    ax.set_title(" GLMM coefficient values \nfor 'Location (Reference: -45)' Interactions")
    
def plotValidationForGLMM_spikeData (eng, featureType, ax):
    df = generateDataForGLMM_spikeData(eng, featureType)
        # Fit the full model
    full_model = smf.glm("spike_rate ~ C(Location, Treatment(-45)) * TrainingGroup * Field", 
                            df, groups=df["Unit"]).fit(reml=False)
    # Fit reduced models with different combinations of variables and interactions
    reduced_model_1 = smf.glm("spike_rate ~ C(Location) * TrainingGroup", 
                                df, family=sm.families.Poisson(), groups=df["Unit"]).fit(reml=False)
    reduced_model_2 = smf.glm("spike_rate ~ C(Location) * Field", 
                                df, family=sm.families.Poisson(),groups=df["Unit"]).fit(reml=False)
    reduced_model_3 = smf.glm("spike_rate ~ C(Location) + TrainingGroup + Field", 
                                df, family=sm.families.Poisson(),groups=df["Unit"]).fit(reml=False)
    reduced_model_4 = smf.glm("spike_rate ~ C(Location) * Field + TrainingGroup", 
                                df, family=sm.families.Poisson(),groups=df["Unit"]).fit(reml=False)
    reduced_model_5 = smf.glm("spike_rate ~ C(Location) + Field * TrainingGroup", 
                                df, family=sm.families.Poisson(),groups=df["Unit"]).fit(reml=False)

    # Collect AIC for each model
    aic_values = [full_model.aic, reduced_model_1.aic, reduced_model_2.aic, 
                reduced_model_3.aic, reduced_model_4.aic,  reduced_model_5.aic]
    aic_values = aic_values - full_model.aic
    model_names = ['Full Model', 'Full Model Without Field', 'Full Model Without Training Group',
                 'Full Model Without Interactions','Full Model With Only Field Interactions',
                 'Full Model With Only Training Group Interactions']

    # Plot AIC values
    ax.barh(model_names, aic_values, color='k')
    ax.set_xlabelxlabel('AIC')
    ax.title('AIC values reference to Full Model ')
    ax.xticks(rotation=45)
    ax.show()
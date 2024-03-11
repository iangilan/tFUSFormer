import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

# Function to load IoU scores from a file
def load_iou_scores(file_path):
    with open(file_path, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
    return np.array(scores)

# Paths to the IoU score files
file_paths = {
    "Interp": "/home/mws/tFUSFormer/test_results/interpolation_unseen2/IoU_vec.txt",
    "FSRCNN": "/home/mws/tFUSFormer/test_results/model_FSRCNN_1ch_unseen2/IoU_vec.txt",
    "SRGAN": "/home/mws/tFUSFormer/test_results/model_SRGAN_1ch_unseen2/IoU_vec.txt",    
    "SESRResNet": "/home/mws/tFUSFormer/test_results/model_SESRResNet_1ch_unseen2/IoU_vec.txt",
    "tFUSFormer_1ch": "/home/mws/tFUSFormer/test_results/model_tFUSFormer_1ch_unseen2/IoU_vec.txt",
    "tFUSFormer_5ch": "/home/mws/tFUSFormer/test_results/model_tFUSFormer_5ch_unseen2/IoU_vec.txt"
}

# Load IoU scores for each model
iou_scores = {model: load_iou_scores(path) for model, path in file_paths.items()}

# Combine into a DataFrame
df = pd.DataFrame({
    "IoU": np.concatenate(list(iou_scores.values())),
    "Model": np.concatenate([[model]*len(scores) for model, scores in iou_scores.items()])
})

# Perform ANOVA
model_ols = ols('IoU ~ C(Model)', data=df).fit()
anova_results = sm.stats.anova_lm(model_ols, typ=2)
print('============================================================================')
print(anova_results)
print('============================================================================')
# Perform Tukey's HSD test
tukey_hsd_result = pairwise_tukeyhsd(endog=df['IoU'], groups=df['Model'], alpha=0.05)

# Print Tukey's HSD test results
print(tukey_hsd_result)
print('============================================================================')

tukey_summary_df = pd.DataFrame(data=tukey_hsd_result.summary().data[1:], columns=tukey_hsd_result.summary().data[0])
print('============================================================================')
print(tukey_summary_df)
print('============================================================================')
#for model, scores in iou_scores.items():
#    print(f"{model}: Mean IoU = {np.mean(scores)}")
#print('============================================================================')    
for model, scores in iou_scores.items():
    mean_iou = np.mean(scores)
    std_dev_iou = np.std(scores)
    print(f"{model}: Mean IoU = {mean_iou:.4f}, Std. Dev. = {std_dev_iou:.4f}")

#============================================================================================
# Boxplot with ANOVA
#============================================================================================ 
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 14  # Legend font size 
    
# Convert model names into a categorical type for better plotting
df['Model'] = pd.Categorical(df['Model'], categories=iou_scores.keys(), ordered=True)

# Convert IoU scores to percentages by multiplying by 100
df['IoU'] = df['IoU'] * 100

model_order = ["Interp", "FSRCNN", "SRGAN", "SESRResNet", "tFUSFormer_1ch", "tFUSFormer_5ch"]

plt.figure(figsize=(8, 12))

palette = sns.color_palette("Set3", n_colors=len(df['Model'].unique()))
custom_props = dict(linewidth=1.5, color='black')
ax = sns.boxplot(x='Model', y='IoU', data=df, palette=palette, width=1, order=model_order,
                 boxprops={'edgecolor': 'black'},
                 whiskerprops=custom_props,
                 capprops=custom_props,
                 medianprops=custom_props)
ax.set_ylim(30, 113)  # Adjust based on your data
ax.set_yticks(range(30, 101, 10))

# Now directly specify 'tFUSFormer_5ch' position based on the updated 'model_order'
tFUSFormer_5ch_pos = model_order.index('tFUSFormer_5ch')

# Filter for significant differences involving 'tFUSFormer_5ch'
significant_pairs = tukey_summary_df[
    (tukey_summary_df['reject']) & 
    ((tukey_summary_df['group1'] == 'tFUSFormer_5ch') | (tukey_summary_df['group2'] == 'tFUSFormer_5ch'))
]

# Set the y position for the annotation lines
y_max = df['IoU'].max()
y_pos_annotation = y_max + 2  # Adjust as needed

# Create a copy of the filtered DataFrame to safely modify it
significant_pairs = tukey_summary_df[
    (tukey_summary_df['reject']) & 
    ((tukey_summary_df['group1'] == 'tFUSFormer_5ch') | (tukey_summary_df['group2'] == 'tFUSFormer_5ch'))
].copy()

# Add a 'distance' column to the DataFrame without triggering the warning
significant_pairs['distance'] = significant_pairs.apply(
    lambda row: abs(model_order.index(row['group1']) - model_order.index(row['group2'])),
    axis=1
)

# Then proceed with sorting and annotations as before
significant_pairs_sorted = significant_pairs.sort_values(by='distance', ascending=False)

for index, row in significant_pairs_sorted.iterrows():
    if row['reject'] and 'tFUSFormer_5ch' in [row['group1'], row['group2']]:  # Check for significance and involvement of 'tFUSFormer_5ch'
        group1_pos = model_order.index(row['group1'])
        group2_pos = model_order.index(row['group2'])
        
        # Draw the annotation line
        plt.plot([group1_pos, group1_pos, group2_pos, group2_pos], [y_pos_annotation, y_pos_annotation + 1, y_pos_annotation + 1, y_pos_annotation], lw=1.5, c='black')
        plt.text((group1_pos + group2_pos) * .5, y_pos_annotation + 0.1, "*", ha='center', va='bottom', color='black', fontsize=20)
        
        # Decrement y_pos_annotation for the next pair to place it lower
        y_pos_annotation += 3  # Adjust spacing as needed


ax.set_title('Comparison of IoU Scores Across Models')
ax.set_ylabel('IoU Score (%)')
ax.set_xlabel('')

# Set custom tick labels for the x-axis
model_labels = ['a', 'b', 'c', 'd', 'e', 'f']  # Letters for each model
ax.set_xticklabels(model_labels)

# Define the model names that correspond to each letter
model_names = [
    'Interpolation', 
    'FSRCNN', 
    'SRGAN', 
    'SE-SRResNet', 
    '1ch-tFUSFormer', 
    '5ch-tFUSFormer'
]


# Create legend patches
legend_patches = [mpatches.Patch(color=palette[i], label=f'{label}. {name}') 
                  for i, (label, name) in enumerate(zip(model_labels, model_names))]

# Create custom legend
#legend_patches = [Patch(color='none', label=f'{letter}. {name}') for letter, name in zip(model_labels, model_names)]
ax.legend(handles=legend_patches, loc='lower right', frameon=True, fontsize='x-large', 
           handlelength=1, handleheight=1, borderpad=1, labelspacing=1)

# Add custom grid lines within y=30 to y=100
for y in np.arange(30, 101, 10):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  

plt.tight_layout()

plt.savefig('iou_scores_comparison_ANOVA_unseen2.png', dpi=600)
plt.show()


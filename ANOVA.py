import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Function to load IoU scores from a file
def load_iou_scores(file_path):
    with open(file_path, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
    return np.array(scores)

# Paths to the IoU score files
file_paths = {
    "Interp": "/home/mws/tFUSFormer/test_results/interpolation_seen/IoU_vec.txt",
    "FSRCNN": "/home/mws/tFUSFormer/test_results/model_FSRCNN_1ch_seen/IoU_vec.txt",
    "SRGAN": "/home/mws/tFUSFormer/test_results/model_SRGAN_1ch_seen/IoU_vec.txt",    
    "SESRResNet": "/home/mws/tFUSFormer/test_results/model_SESRResNet_1ch_seen/IoU_vec.txt",
    "tFUSFormer_1ch": "/home/mws/tFUSFormer/test_results/model_tFUSFormer_1ch_seen/IoU_vec.txt",
    "tFUSFormer_5ch": "/home/mws/tFUSFormer/test_results/model_tFUSFormer_5ch_seen/IoU_vec.txt"
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
 
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 14  # Legend font size 
    
# Convert model names into a categorical type for better plotting
df['Model'] = pd.Categorical(df['Model'], categories=iou_scores.keys(), ordered=True)

# Convert IoU scores to percentages by multiplying by 100
df['IoU'] = df['IoU'] * 100
    
# Create a boxplot
plt.figure(figsize=(8, 12))
sns.boxplot(x='Model', y='IoU', data=df, palette="Set3", width=1)  # Increase width to reduce gaps
plt.title('Comparison of IoU Scores Across Models')
plt.ylabel('IoU Score (%)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('iou_scores_comparison.png', dpi=600)
plt.show()



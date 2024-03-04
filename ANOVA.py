import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to load IoU scores from a file
def load_iou_scores(file_path):
    with open(file_path, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
    return np.array(scores)

# Paths to the IoU score files
file_paths = {
    "FSRCNN": "/home/mws/tFUSFormer/test_results/model_FSRCNN_1ch_seen/IoU_vec.txt",
    "SESRResNet": "/home/mws/tFUSFormer/test_results/model_SESRResNet_1ch_seen/IoU_vec.txt",
    "SRGAN": "/home/mws/tFUSFormer/test_results/model_SRGAN_1ch_seen/IoU_vec.txt",
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
print('======================================================')
print(anova_results)

# Perform Tukey's HSD test
tukey_hsd_result = pairwise_tukeyhsd(endog=df['IoU'], groups=df['Model'], alpha=0.05)

# Print Tukey's HSD test results
#print(tukey_hsd_result)


tukey_summary_df = pd.DataFrame(data=tukey_hsd_result.summary().data[1:], columns=tukey_hsd_result.summary().data[0])
print('===========================================================================')
print(tukey_summary_df)

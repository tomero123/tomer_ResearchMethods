import pandas as pd
from scipy.stats import ttest_ind

path = "C:\\university\\Semester C\\Research Methods\\Excercises\\Ex4\\Results\\"
antibiotic = "isoniazid"
file_name = f"{antibiotic}_FeatureSelection_folds_results.csv"

df = pd.read_csv(path + file_name)
fs_th_list = list(df['Feature Selection #'].unique())

results = []

for i in range(len(fs_th_list)):
    for j in range(i + 1, len(fs_th_list)):
        algo_1_folds = df[df['Feature Selection #'] == fs_th_list[i]]
        algo_2_folds = df[df['Feature Selection #'] == fs_th_list[j]]
        accuracy_t_test = ttest_ind(algo_1_folds['accuracy'], algo_2_folds['accuracy'])
        f1_score_t_test = ttest_ind(algo_1_folds['f1_score'], algo_2_folds['f1_score'])
        results.append([fs_th_list[i], fs_th_list[j], accuracy_t_test[0], accuracy_t_test[1], f1_score_t_test[0], f1_score_t_test[1]])

results_df = pd.DataFrame(results, columns=["Feature Selection # 1", "Feature Selection # 2", "Accuracy t-statistic", "Accuracy p-value", "F1-score t-statistic", "F1-score p-value"])
results_df.to_csv(path + f"{antibiotic}_FS_NUMBER_t_test.csv", index=False)

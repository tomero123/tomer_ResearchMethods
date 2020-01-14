import pandas as pd
from scipy.stats import ttest_ind

path = "C:\\university\\Semester C\\Research Methods\\Excercises\\Ex4\\Results\\"
antibiotic = "ethambutol"
file_name = f"{antibiotic}_folds_results.csv"

df = pd.read_csv(path + file_name)
algorithms = list(df['Model'].unique())

results = []

for i in range(len(algorithms)):
    for j in range(i + 1, len(algorithms)):
        algo_1_folds = df[df['Model'] == algorithms[i]]
        algo_2_folds = df[df['Model'] == algorithms[j]]
        accuracy_t_test = ttest_ind(algo_1_folds['accuracy'], algo_2_folds['accuracy'])
        f1_score_t_test = ttest_ind(algo_1_folds['f1_score'], algo_2_folds['f1_score'])
        results.append([algorithms[i], algorithms[j], accuracy_t_test[0], accuracy_t_test[1], f1_score_t_test[0], f1_score_t_test[1]])

results_df = pd.DataFrame(results, columns=["Model 1", "Model 2", "Accuracy t-statistic", "Accuracy p-value", "F1-score t-statistic", "F1-score p-value"])
results_df.to_csv(path + f"{antibiotic}_t_test.csv", index=False)

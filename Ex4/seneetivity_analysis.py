import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

path = "C:\\university\\Semester C\\Research Methods\\Excercises\\Ex4\\CV_Results_20112019_FS_300\\"
antibiotic = "isoniazid"
random_seed = 1
k_folds = 10
num_of_processes = 1


file_name = f"{antibiotic}_FS_DATA.csv"
final_df = pd.read_csv(path + file_name)
feature_selection_list = [100, 300, 500, 1000]
fig = plt.figure(figsize=(8, 6))

# models_list = [xgboost.XGBClassifier(random_state=random_seed)]
folds_results = []
for fs_th in feature_selection_list:
    # Replace R->1, S->0
    X = final_df.drop(['label'], axis=1).copy()
    y = final_df[['label']].copy()
    y = y.replace("R", 1).replace("S", 0)
    model = xgboost.XGBClassifier(random_state=random_seed)
    model_name = type(model).__name__
    # model.set_params(**model_params)

    if fs_th < 1000:
        # X = SelectKBest(chi2, k=features_selection_n).fit_transform(X, y)
        print(f"Started Feature selection model fit antibiotic: {antibiotic}")
        now = time.time()
        model.fit(X, y.values.ravel())
        # Write csv of data after FS
        d = model.feature_importances_
        most_important_index = sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:fs_th]
        temp_df = X.iloc[:, most_important_index]
        selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=fs_th)
        X = selection.transform(X)

    model = xgboost.XGBClassifier(random_state=random_seed)
    cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
    print("Started running Cross Validation for {} folds with {} processes".format(k_folds, num_of_processes))
    now = time.time()
    classes = np.unique(y.values.ravel())
    temp_scores = cross_val_predict(model, X, y.values.ravel(), cv=cv,
                                    method='predict_proba', n_jobs=num_of_processes)
    predictions = np.array(temp_scores[:, 1])
    y_pred = np.array([1 if i > 0.5 else 0 for i in predictions])
    y_true = np.array(y['label'])
    index_list_temp = np.array((range(len(predictions))))

    fpr, tpr, _ = roc_curve(y_true,  predictions)
    auc = roc_auc_score(y_true, predictions)

    plt.plot(fpr, tpr, label="FS threshold {}, AUC={:.3f}".format(fs_th, auc))

    np.random.shuffle(index_list_temp)
    index_list = np.array_split(index_list_temp, k_folds)
    for j, ind in enumerate(index_list):
        cur_y_true, cur_y_pred = y_true[ind], y_pred[ind]
        accuracy = metrics.accuracy_score(cur_y_true, cur_y_pred)
        f1_score = metrics.f1_score(cur_y_true, cur_y_pred)
        folds_results.append([model_name, fs_th, j + 1, accuracy, f1_score])

    print("Finished running Model: {} with fs_th: {}".format(model_name, fs_th))


plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title(f'ROC Curve Analysis for antibiotic {antibiotic}', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.savefig(path + f"{antibiotic}_FeatureSelection_ROC.png")

results_df = pd.DataFrame(folds_results, columns=["Model", "Feature Selection #", "Fold", "accuracy", "f1_score"])
results_df.to_csv(path + f"{antibiotic}_FeatureSelection_folds_results.csv", index=False)

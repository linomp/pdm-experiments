# %%
import os
import time
from collections import Counter

import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.cleaning import visualize_missing_values, drop_cols_with_quality_threshold, get_snake_case_column_mapping
from utils.evaluation import eval_classifier
from utils.training import xgb_build

MODELS_DIR = './models'
RANDOM_STATE = 0

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

# %%
# Load & Clean Data
df_base = pd.read_csv("data/milling_synthetic_kaggle.csv")
df_base.drop(columns=['Target', 'Product ID', 'UDI'], inplace=True)

print(df_base.shape)

if df_base.isna().sum().sum() > 0:
    visualize_missing_values(df_base)
    df_base = drop_cols_with_quality_threshold(df_base, 0.05)
else:
    print('no missing values')

df_base.rename(columns=get_snake_case_column_mapping(df_base.columns), inplace=True)

target_name = 'failure_type'
categorical_columns = ["type", "failure_type"]

df_base = df_base.dropna()
class_mapping = {
    'No Failure': 0,
    'Power Failure': 1,
    'Tool Wear Failure': 2,
    'Overstrain Failure': 3,
    'Random Failures': 4,
    'Heat Dissipation Failure': 5
}
df_base[target_name].replace(class_mapping, inplace=True)
df_base['type'].replace({'L': 0, 'M': 1, 'H': 2}, inplace=True)

X = df_base.drop(columns=[target_name])
y = df_base[target_name]

print(f"Columns used for training: {X.columns}")

# %%
# Split & Handle class imbalance
X, X_validate, y, y_validate = train_test_split(X, y, train_size=0.7, random_state=RANDOM_STATE, stratify=y)

X_resampled, y_resampled = ADASYN(random_state=RANDOM_STATE).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.7, random_state=RANDOM_STATE,
                                                    stratify=y_resampled)

print('\nDataset shapes')
print("Class counts[original]: ", Counter(y))
print("Class counts[ADASYN]: ", Counter(y_resampled))
print('train[ADASYN]: ', X_train.shape, y_train.shape, Counter(y_train))
print('test[ADASYN]: ', X_test.shape, y_test.shape, Counter(y_test))
print('validate[ADASYN]: ', X_validate.shape, y_validate.shape, Counter(y_validate))

# %%
# Train Models

# Decision Tree
start = time.time()
dt_adasyn_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_adasyn_clf.fit(X_train, y_train)
end = time.time()

report = eval_classifier(dt_adasyn_clf, X_validate, y_validate, X_train, y_train, show_confusion_matrix=False,
                         class_mapping=class_mapping, fig_title='Decision Tree [ADASYN]')

print(f"\nTraining time [Decision Tree]: {end - start}s")
print(report)

# Random Forests
start = time.time()
rf_adasyn_clf = RandomForestClassifier(random_state=RANDOM_STATE)
rf_adasyn_clf.fit(X_train, y_train)
end = time.time()

report = eval_classifier(rf_adasyn_clf, X_validate, y_validate, X_train, y_train, show_confusion_matrix=False,
                         class_mapping=class_mapping, fig_title='Random Forests [ADASYN]')

print(f"\nTraining time [Random Forests]: {end - start}s")
print(report)

# XGBoost
start = time.time()
xgb_adasyn_clf = xgb_build(X_train, y_train, X_test, y_test)
end = time.time()

report = eval_classifier(xgb_adasyn_clf, X_validate, y_validate, X_train, y_train, show_confusion_matrix=False,
                         class_mapping=class_mapping, fig_title='XGBoost [ADASYN]')

print(f"\nTraining time [XGBoost]: {end - start}s")
print(report)

# %%
plt.show(block=True)

# %%
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from utils.cleaning import visualize_missing_values, drop_cols_with_quality_threshold, get_snake_case_column_mapping
from utils.evaluation import eval_classifier
from utils.training import xgb_build_eval, xgb_build
from utils.visualization import get_feature_boxplots

MODELS_DIR = 'models'
RANDOM_STATE = 0

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

# %%
# 1. Load and clean the data

df_base = pd.read_csv("data/milling_synthetic_kaggle.csv")

# The 'Target' column indicates whether the machine failed or not. We drop to avoid leakage
# We also drop the unnecessary ID columns
df_base.drop(columns=['Target', 'Product ID', 'UDI'], inplace=True)

print(df_base.shape)
print(df_base.head(3))

# Examine missing value and drop all columns with more than 5% missing values
if df_base.isna().sum().sum() > 0:
    visualize_missing_values(df_base)
    df_base = drop_cols_with_quality_threshold(df_base, 0.05)
else:
    print('no missing values')

# Format column names to snake case
df_base.rename(columns=get_snake_case_column_mapping(df_base.columns), inplace=True)
print(f"Columns after cleaning: {df_base.columns}")

# We want to predict the type of failure
target_name = 'failure_type'
categorical_columns = ["type", "failure_type"]

# %%
# 2. Exploratory Data Analysis
if not os.getenv('SKIP_EDA', False):
    # 2.1 Class (im)balance
    ax = sns.countplot(y=target_name, data=df_base, order=df_base[target_name].value_counts().index)
    ax.figure.tight_layout()
    plt.show(block=False)

    # 2.2 Pair-plots to explore relationships between features, colored by Failure Type
    sns.pairplot(df_base, height=2.5, hue='failure_type')
    plt.show(block=False)

    # 2.3 Correlation matrix between features
    ax = sns.heatmap(df_base.drop(columns=categorical_columns).corr(), cbar=True, fmt='.1f', vmax=0.8, annot=True,
                     cmap='Blues')
    ax.figure.tight_layout()
    plt.show(block=False)

    # 2.4 Feature boxplots colored by Failure Type
    get_feature_boxplots(df_base, 'air_temperature_k').show()
    get_feature_boxplots(df_base, 'process_temperature_k').show()
    get_feature_boxplots(df_base, 'rotational_speed_rpm').show()
    get_feature_boxplots(df_base, 'torque_nm').show()

# %%
# 3. Data preparation (encode categorical features, split into train/test sets)
df_base = df_base.dropna()

# Specific column mappings: failure types and type of piece being worked (quality level)
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

# %%
# 4. Model Training & Evaluation

X, X_validate, y, y_validate = train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=RANDOM_STATE, stratify=y)

print('\n\nRaw results - no class imbalance handling')
print('Class counts:', Counter(y))
print('train[raw]: ', X_train.shape, y_train.shape)
print('test[raw]: ', X_test.shape, y_test.shape)
print('validate[raw]: ', X_validate.shape, y_validate.shape)

clf, report = xgb_build_eval(X_train, y_train, X_test, y_test)
print("XGBoost[raw] scores:\n", report)

# %%
# 5. Improvements: handle class imbalance problem

# 5.1. Downsampling

X_resampled, y_resampled = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.7, random_state=RANDOM_STATE,
                                                    stratify=y_resampled)

print('\n\nDownsampling results')
print('Class counts:', Counter(y_resampled))
print('train[downsampling]: ', X_train.shape, y_train.shape)
print('test[downsampling]: ', X_test.shape, y_test.shape)

clf_under = xgb_build(X_train, y_train, X_test, y_test)
print("XGBoost[downsampling] validation set scores:\n",
      eval_classifier(clf_under, X_validate, y_validate, class_mapping=class_mapping, show_confusion_matrix=True,
                      fig_title='Downsampling'))

# 5.2. SMOTE

X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.7, random_state=RANDOM_STATE,
                                                    stratify=y_resampled)

print('\n\nSMOTE results')
print('Class counts:', Counter(y_resampled))
print('train[SMOTE]: ', X_train.shape, y_train.shape)
print('test[SMOTE]: ', X_test.shape, y_test.shape)

clf_smote = xgb_build(X_train, y_train, X_test, y_test)
print("XGBoost[SMOTE] validation set scores:\n",
      eval_classifier(clf_smote, X_validate, y_validate, class_mapping=class_mapping, show_confusion_matrix=True,
                      fig_title='SMOTE'))

# 5.3. ADASYN

X_resampled, y_resampled = ADASYN(random_state=RANDOM_STATE).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.7, random_state=RANDOM_STATE,
                                                    stratify=y_resampled)

print('\n\nADASYN results')
print('Class counts:', Counter(y_resampled))
print('train[ADASYN]: ', X_train.shape, y_train.shape)
print('test[ADASYN]: ', X_test.shape, y_test.shape)

clf_adasyn = xgb_build(X_train, y_train, X_test, y_test)
print("XGBoost[ADASYN] validation set scores:\n",
      eval_classifier(clf_adasyn, X_validate, y_validate, class_mapping=class_mapping, show_confusion_matrix=True,
                      fig_title='ADASYN'))

# %%
# 6. Save the best model
pickle.dump(clf_adasyn, open(f"{MODELS_DIR}/xgboost_milling_multiclass.pkl", "wb"))

plt.show(block=True)

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wittgenstein
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

from utils.cleaning import visualize_missing_values, drop_cols_with_quality_threshold, get_snake_case_column_mapping
from utils.evaluation import eval_classifier
from utils.visualization import get_feature_boxplots

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

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

# 3. Data preparation (encode categorical features, split into train/test sets)
df_base = df_base.dropna()

# TODO: handle class imbalance problem. downsapling, oversampling, what else exists?
# df_base = get_downsampled_df(df_base, target_name)
# print(df_base[target_name].value_counts())

# Specific column mappings: failure types and type of piece being worked (quality level)
df_base[target_name].replace(
    {
        'No Failure': 0,
        'Power Failure': 1,
        'Tool Wear Failure': 2,
        'Overstrain Failure': 3,
        'Random Failures': 4,
        'Heat Dissipation Failure': 5
    }, inplace=True
)
df_base['type'].replace({'L': 0, 'M': 1, 'H': 2}, inplace=True)

y = df_base[target_name]
X = df_base.drop(columns=[target_name])

# TODO: and the validation set?
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

print('train: ', X_train.shape, y_train.shape)
print('test: ', X_test.shape, y_test.shape)

# 4. Model Training & Evaluation

# 4.1 XGBoost
weight_train = compute_sample_weight('balanced', y_train)
weight_test = compute_sample_weight('balanced', y_test)
xgb_clf = XGBClassifier(booster='gbtree',
                        tree_method='gpu_hist',
                        sampling_method='gradient_based',
                        eval_metric='aucpr',
                        objective='multi:softmax',
                        num_class=6)
xgb_clf.fit(X_train, y_train.ravel(), sample_weight=weight_train)

eval_classifier(xgb_clf, X_test, y_test, X_train, y_train, do_cross_validation=False, weight_train=weight_train,
                weight_test=weight_test)

# 4.2 Wittgenstein
ripper_clf = wittgenstein.RIPPER()

# Wittgenstein is not for multiclass classification, so we need to drop the other classes.
# For now we keep only the 'No Failure' and 'Power Failure' classes (by now mapped to 0 and 1)
df_base = df_base[df_base[target_name].isin([0, 1])]

ripper_clf.fit(df_base, class_feat=target_name, pos_class=0)

print(ripper_clf.out_model())

# Recompute train/test sets for binary classification
y = df_base[target_name]
X = df_base.drop(columns=[target_name])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

eval_classifier(ripper_clf, X_test, y_test, X_train, y_train, do_cross_validation=False)

# TODO: how to handle undersampling and test set? (confusion matrix gets generated with very few samples)
# TODO: test with wittgenstein, CatBoost and simpler models

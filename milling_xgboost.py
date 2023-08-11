import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support as score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

from utils.cleaning import visualize_missing_values, drop_cols_with_quality_threshold, get_snake_case_column_mapping
from utils.viz import get_feature_boxplots

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

# 2. Exploratory Data Analysis

# 2.1 Class (im)balance
ax = sns.countplot(y=target_name, data=df_base, order=df_base[target_name].value_counts().index)
ax.figure.tight_layout()
plt.show(block=False)

# 2.2 Pair-plots to explore relationships between features, colored by Failure Type
sns.pairplot(df_base, height=2.5, hue='failure_type')
plt.show(block=False)

# 2.3 Correlation matrix between features
ax = sns.heatmap(df_base.drop(columns=["type", "failure_type"]).corr(), cbar=True, fmt='.1f', vmax=0.8, annot=True,
                 cmap='Blues')
ax.figure.tight_layout()
plt.show(block=False)

# 2.4 Feature boxplots colored by Failure Type
get_feature_boxplots(df_base, 'air_temperature_k').show()
get_feature_boxplots(df_base, 'process_temperature_k').show()
get_feature_boxplots(df_base, 'rotational_speed_rpm').show()
get_feature_boxplots(df_base, 'torque_nm').show()

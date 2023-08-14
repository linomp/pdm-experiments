import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame


def visualize_missing_values(df: DataFrame, do_plot=True) -> DataFrame:
    null_df = pd.DataFrame(df.isna().sum(), columns=['null_values']).sort_values(['null_values'], ascending=False)

    if do_plot:
        fig = plt.subplots(figsize=(16, 6))
        ax = sns.barplot(data=null_df, x='null_values', y=null_df.index, color='royalblue')
        pct_values = [' {:g}'.format(elm) + ' ({:.1%})'.format(elm / len(df)) for elm in list(null_df['null_values'])]
        ax.set_title('Overview of missing values')
        ax.bar_label(container=ax.containers[0], labels=pct_values, size=12)
        plt.show()

    return null_df


def drop_cols_with_quality_threshold(df: DataFrame, threshold=0.05):
    original_cols = df.columns
    for col_name in original_cols:
        if df[col_name].isna().sum() / df.shape[0] > threshold:
            df = df.drop(columns=[col_name])

    return df


def get_snake_case_column_mapping(column_names: list[str]):
    snake_case_mapping = {}

    for col in column_names:
        snake_case_mapping[col] = re.sub(r'\W+', '_', col).strip('_').lower()

    return snake_case_mapping


#
# def get_downsampled_df(df_base: DataFrame, target_name: str):
#     # process the DF as to have a balanced dataset, by taking the same number of samples from each class
#
#     minority_class_count = df_base[target_name].value_counts().min()
#
#     df_downsampled = pd.DataFrame()
#     for failure_type in df_base[target_name].unique():
#         df_downsampled = pd.concat(
#             [df_downsampled,
#              df_base[df_base[target_name] == failure_type].sample(n=minority_class_count, random_state=0)]
#         )
#
#     return df_downsampled


def get_downsampled_df(df_base: pd.DataFrame, target_name: str):
    X = df_base.drop(columns=[target_name])
    y = df_base[target_name]

    # Create an instance of RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=0)

    # Resample the dataset
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Combine the resampled features and target into a DataFrame
    df_downsampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_name)],
                               axis=1)

    return df_downsampled


def get_oversampled_df(df_base: pd.DataFrame, target_name: str):
    # TODO: check if this is the right way to do it. There may be leakage between train and test sets

    X = df_base.drop(columns=[target_name])
    y = df_base[target_name]

    # Create an instance of RandomOverSampler
    ros = RandomOverSampler(sampling_strategy='auto', random_state=0)

    # Resample the dataset
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Combine the resampled features and target into a DataFrame
    df_oversampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_name)],
                               axis=1)

    return df_oversampled

import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

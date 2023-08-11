from matplotlib import pyplot as plt
from pandas import DataFrame
import plotly.express as px


def get_feature_boxplots(df: DataFrame, column_name: str, target_name: str = 'failure_type'):
    plt.figure(figsize=(16, 6))
    return px.box(data_frame=df, y=column_name, color=target_name, points="all", width=1200)

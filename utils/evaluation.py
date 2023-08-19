import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate


def eval_classifier(clf, x_test, y_test, x_train=None, y_train=None, do_cross_validation=False,
                    show_confusion_matrix=False, class_mapping: dict | None = None,
                    fig_title: str | None = None) -> str:
    # Binary Classification Report
    y_pred = clf.predict(x_test)
    target_names = None
    if class_mapping is not None:
        # Getting the labels like this works because Python 3.7 guarantees that the order of insertion is preserved
        target_names = list(class_mapping.keys())
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0, target_names=target_names)

    # Multi-class Confusion Matrix
    if show_confusion_matrix:
        cnf_matrix = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cnf_matrix, columns=np.unique(y_test), index=np.unique(y_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(8, 5))
        plt.title(fig_title)
        sns.set(font_scale=1.1)  # for label size
        sns.heatmap(df_cm, cbar=True, cmap="inferno", annot=True, fmt='.0f')
        plt.show(block=False)

    if do_cross_validation and (x_train is not None) and (y_train is not None):
        scores = cross_validate(clf, x_train, y_train, cv=10, scoring="f1_weighted")
        scores_df = pd.DataFrame(scores)
        px.bar(x=scores_df.index, y=scores_df.test_score, width=800).show()

    return report

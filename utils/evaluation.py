import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate


def eval_classifier(clf, x_test, y_test, x_train, y_train, do_cross_validation=False, weight_train=None,
                    weight_test=None):
    # score = clf.score(x_test, y_test.ravel(), sample_weight=weight_test)
    y_pred = clf.predict(x_test)

    # Binary Classification Report
    print(classification_report(y_test, y_pred))

    # Multi-class Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cnf_matrix, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(8, 5))
    sns.set(font_scale=1.1)  # for label size
    sns.heatmap(df_cm, cbar=True, cmap="inferno", annot=True, fmt='.0f')
    plt.show()

    if do_cross_validation:
        scores = cross_validate(clf, x_train, y_train, cv=10, scoring="f1_weighted",
                                fit_params={"sample_weight": weight_train})
        scores_df = pd.DataFrame(scores)
        px.bar(x=scores_df.index, y=scores_df.test_score, width=800).show()

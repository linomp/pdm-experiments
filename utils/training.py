import numpy as np
from xgboost import XGBClassifier

from utils.evaluation import eval_classifier


def xgb_build(_x_train, _y_train, _x_test, _y_test) -> XGBClassifier:
    # TODO: make tree method dependent on GPU availability
    # TODO: num_class is currently hardcoded
    xgb_clf = XGBClassifier(
        booster='gbtree',
        tree_method='gpu_hist',
        sampling_method='gradient_based',
        eval_metric='aucpr',
        objective='multi:softmax',
        num_class=len(np.unique(_y_train))
    )
    xgb_clf.fit(_x_train, _y_train.ravel())

    return xgb_clf


def xgb_build_eval(_x_train, _y_train, _x_test, _y_test, class_mapping: dict | None = None, do_cross_validation=False,
                   show_confusion_matrix=False) -> \
        tuple[XGBClassifier, str]:
    xgb_clf = xgb_build(_x_train, _y_train, _x_test, _y_test)
    return xgb_clf, eval_classifier(xgb_clf, _x_test, _y_test, _x_train, _y_train,
                                    do_cross_validation=do_cross_validation,
                                    show_confusion_matrix=show_confusion_matrix,
                                    class_mapping=class_mapping)

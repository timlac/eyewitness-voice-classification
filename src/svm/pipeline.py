from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

from src.plots.auc import plot_auc_curve
from src.plots.misc import plot_hist, plot_conf_mat
from src.svm.param_search import param_search
from src.utils import evaluate_scores, get_splits


def svm_pipeline(x, y, scoring_method):
    """
    Performs grid search
    Evaluates scores with cross validation, using classifier with the best parameters
    Visualizes results with confusion matrix, AUC curve etc.

    :param x: input data
    :param y: output data (ground truth)
    :param scoring_method: The best parameters will be selected based on this scoring method
    e.g. f1_macro, roc_auc, cohen_kappa_score
    """

    clf = SVC()

    # do grid search for best parameters
    gs = param_search(x, y, clf, scoring_method)

    svc = SVC(**gs.best_params_)

    # evaluate classifier with different scoring methods
    # check cohens kappa also
    evaluate_scores(x, y, svc, "accuracy")
    evaluate_scores(x, y, svc, "roc_auc")
    evaluate_scores(x, y, svc, "f1_macro")

    # get the predictions using cross validation
    splits = get_splits(x, y)
    y_pred = cross_val_predict(svc, x, y, cv=splits, n_jobs=-1)

    # plot histogram for predictions to show distribution
    plot_hist(y_pred, "predictions")

    # get classification report based on predictions
    report = metrics.classification_report(y_true=y, y_pred=y_pred)
    print(report)

    # plot confusion matrix
    conf_mat = confusion_matrix(y, y_pred)
    plot_conf_mat(conf_mat)
    print(conf_mat)

    # plot auc curve
    # plot precision recall curve
    svc = SVC(**gs.best_params_, probability=True)

    print("params after probability set to True:")
    print(svc.get_params())

    plot_auc_curve(svc, x, y)

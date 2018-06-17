from models.baseline import BaselineGender


def evaluate_model(model, X_eval, y_eval):
    """
    :param model:
    :param X_eval:
    :param y_eval:
    :return:
    """
    print("> Evaluating model...")
    print("Test set size: %s" % str(X_eval.shape))

    # Baseline is the simplest model that is easy to see
    baseline_model = BaselineGender()
    baseline_score = baseline_model.score(X_eval, y_eval)

    # we want to see if this score is better than baseline or not
    score = model.score(X_eval, y_eval)

    print("Baseline Accuracy: %s" % baseline_score)
    print("Accuracy: %s" % score)
    print("----------------")


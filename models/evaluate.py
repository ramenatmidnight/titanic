def evaluate_model(model, X_eval, y_eval):
    print("> Evaluating model...")
    print("Test set size: %s" % str(X_eval.shape))
    score = model.score(X_eval, y_eval)
    print(score)
    print("----------------")

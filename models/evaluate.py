def evaluate_model(model, X_test, y_test):
    print("> Evaluating model...")
    print("Test set size: %s" % str(X_test.shape))
    score = model.score(X_test, y_test)
    print("----------------")
    print(score)
